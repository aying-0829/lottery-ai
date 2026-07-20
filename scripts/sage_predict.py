"""
彩圣榜 · 每日推荐存档 — 方案2（真实回测闭环）
每次 daily-update workflow 在 model.py 之后运行：
  - 对双色球 / 大乐透，遍历 18 位历史人物（每人一种策略），
    用与前端一致的「策略引擎」从模型 rf/nb/mc 评分生成推荐组合，
    追加到 data/sage_predict_history.json
  - 对历史中已存在但尚未回填的条目，用 data 中的实际开奖回写命中情况
设计要点：
  - 种子确定性：seed = f(目标期号, sage_id, 彩种)，保证
    「CI存档推荐 == 前端展示 == 回测对象」三者完全一致，可复现。
  - 策略引擎 Python 实现需与 index.html 中 _sagePickNumbers 逐字节一致，
    否则前端展示与回测会分叉。验证脚本见仓库 docs/ 或 CI 日志。
  - 按 (game, sage_id, target_period) 去重，避免重复写入同一期
  - 原子写（先写 .tmp 再 os.replace），避免 CI 并发写入损坏文件
依赖：仅标准库（json / os / math / datetime）。
"""
import json
import os
import math
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TZ = timezone(timedelta(hours=8))

SAGE_PREDICT_FILE = os.path.join(DATA_DIR, "sage_predict_history.json")

# ─── 18 位历史人物（与 index.html SAGES_DATA 一致；此处仅需 id + strategy）───
# id 必须唯一且与前端完全一致；strategy 决定选号风格。
SAGES = [
    {"id": "zhugeliang",   "name": "诸葛亮", "strategy": "balanced"},
    {"id": "jiangziya",    "name": "姜子牙", "strategy": "cold"},
    {"id": "gongsunsheng", "name": "公孙胜", "strategy": "hot"},
    {"id": "chaogai",      "name": "晁盖",   "strategy": "aggressive"},
    {"id": "simaqian",     "name": "司马迁", "strategy": "analysis"},
    {"id": "zongzongtang", "name": "左宗棠", "strategy": "aggressive"},
    {"id": "sushi",        "name": "苏轼",   "strategy": "random"},
    {"id": "lvqinghou",    "name": "吕轻侯", "strategy": "conservative"},
    {"id": "zhangfei",     "name": "张飞",   "strategy": "wild"},
    {"id": "guofurong",    "name": "郭芙蓉", "strategy": "wild"},
    {"id": "taoyuanming",  "name": "陶渊明", "strategy": "random"},
    {"id": "luobinwang",   "name": "骆宾王", "strategy": "balanced"},
    {"id": "zhangqian",    "name": "张骞",   "strategy": "explorer"},
    {"id": "cailun",       "name": "蔡伦",   "strategy": "innovation"},
    {"id": "shenkuo",      "name": "沈括",   "strategy": "science"},
    {"id": "liubei",       "name": "刘备",   "strategy": "kind"},
    {"id": "guanyu",       "name": "关羽",   "strategy": "loyal"},
    {"id": "huajing",      "name": "花荣",   "strategy": "precision"},
]


# ─── 工具 ────────────────────────────────────────────
def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def next_period(period, game):
    """计算下一期期号（与 save_predict.py 完全一致）。"""
    if game == "ssq":
        year = int(period[:4]); seq = int(period[4:])
        if seq >= 999:
            return f"{year + 1}001"
        return f"{year}{seq + 1:03d}"
    else:
        year = int(period[:2]); seq = int(period[2:])
        if seq >= 999:
            return f"{year + 1:02d}001"
        return f"{year:02d}{seq + 1:03d}"


def imul(a, b):
    """JS Math.imul 的 Python 等价实现（32 位有符号乘法，取低 32 位）。"""
    return (a * b) & 0xFFFFFFFF


def sage_seed(target_period, sage_id, game):
    """确定性种子：与 index.html sageSeed() 完全一致。"""
    base = int(target_period)
    sid = sum(ord(c) for c in sage_id)
    goff = 1 if game == "dlt" else 0
    return (base * 31 + sid * 7 + goff * 13) & 0x7FFFFFFF


def _sage_seed_random(seed):
    """确定性伪随机生成器（与 index.html _sageSeedRandom 逐字节一致）。
    返回 [0,1) 浮点序列。"""
    x = seed & 0xFFFFFFFF

    def rng():
        nonlocal x
        x = (x + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(x ^ (x >> 15), (1 | x) & 0xFFFFFFFF)
        t = ((t + imul(t ^ (t >> 7), (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rng


def get_top_from_model(model_data, count):
    """从模型的 rf/nb/mc 评分求和取 Top（mc 权重 0.5），与前端 _sagePickNumbers 一致。"""
    if not model_data or "rf" not in model_data:
        return []
    scores = {}
    for mkey, w in (("rf", 1.0), ("nb", 1.0), ("mc", 0.5)):
        for ball, sc in model_data.get(mkey, []):
            k = str(ball)
            scores[k] = scores.get(k, 0.0) + float(sc) * w
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [b for b, _ in ranked[:count]]


def _safe_get(lst, i):
    try:
        return lst[i]
    except (IndexError, TypeError):
        return None


def sage_pick_numbers(sage, game, model, target_period):
    """Python 版策略引擎，与 index.html _sagePickNumbers 完全一致。
    返回 {"red": [...int], "blue": [...int]} (ssq) 或 {"front":[...], "back":[...]} (dlt)。
    """
    red_key = "red" if game == "ssq" else "front"
    blue_key = "blue" if game == "ssq" else "back"
    red_model = model.get(red_key, {})
    blue_model = model.get(blue_key, {})
    if not red_model or not blue_model:
        return None

    all_red_top = get_top_from_model(red_model, 20)
    all_blue_top = get_top_from_model(blue_model, 8)

    front_count = 6 if game == "ssq" else 5
    back_count = 1 if game == "ssq" else 2
    max_red = 33 if game == "ssq" else 35
    max_blue = 16 if game == "ssq" else 12

    picked_red, picked_blue = [], []
    rng = _sage_seed_random(sage_seed(target_period, sage["id"], game))

    strat = sage["strategy"]
    if strat == "hot":
        picked_red = list(all_red_top[:front_count])
        picked_blue = list(all_blue_top[:back_count])

    elif strat == "cold":
        picked_red = list(all_red_top[-front_count:])
        if len(picked_red) < front_count:
            mid = len(all_red_top) // 2
            extra = all_red_top[mid - 2: mid + 4]
            picked_red = extra[:front_count]
        pb = all_blue_top[-back_count:] if all_blue_top[-back_count:] else all_blue_top[:back_count]
        picked_blue = list(pb)

    elif strat == "balanced":
        mid = len(all_red_top) // 2
        picked_red = [
            _safe_get(all_red_top, 0), _safe_get(all_red_top, 2),
            _safe_get(all_red_top, mid), _safe_get(all_red_top, mid + 2),
            _safe_get(all_red_top, len(all_red_top) - 3),
            _safe_get(all_red_top, len(all_red_top) - 1),
        ]
        picked_red = [x for x in picked_red if x is not None][:front_count]
        midb = len(all_blue_top) // 2
        picked_blue = [
            _safe_get(all_blue_top, 0), _safe_get(all_blue_top, midb)
        ]
        picked_blue = [x for x in picked_blue if x is not None][:back_count]

    elif strat == "aggressive":
        core_red = list(all_red_top[:3])
        picked_red = list(core_red)
        while len(picked_red) < front_count:
            pick = _safe_get(all_red_top, math.floor(rng() * min(10, len(all_red_top))))
            if pick is not None and pick not in picked_red:
                picked_red.append(pick)
        picked_blue = [all_blue_top[0]] if all_blue_top else []
        if back_count == 2 and len(all_blue_top) > 1:
            picked_blue.append(all_blue_top[1])

    elif strat == "conservative":
        mid_start = len(all_red_top) // 2 - 3
        picked_red = list(all_red_top[max(0, mid_start): mid_start + front_count])
        if len(picked_red) < front_count:
            picked_red = list(all_red_top[:front_count])
        midb = len(all_blue_top) // 2
        picked_blue = list(all_blue_top[midb - 1: midb + back_count - 1])
        picked_blue = [x for x in picked_blue if x is not None]
        if len(picked_blue) < back_count:
            picked_blue = list(all_blue_top[:back_count])

    else:  # random / 未识别策略
        pool_red = list(all_red_top[:15])
        while len(picked_red) < front_count and pool_red:
            idx = math.floor(rng() * len(pool_red))
            picked_red.append(pool_red.pop(idx))
        pool_blue = list(all_blue_top[:6])
        while len(picked_blue) < back_count and pool_blue:
            idx2 = math.floor(rng() * len(pool_blue))
            picked_blue.append(pool_blue.pop(idx2))

    # 过滤有效 & 去重
    picked_red = [str(n) for n in picked_red if n is not None and str(n).isdigit()]
    picked_red = list(dict.fromkeys(picked_red))
    picked_blue = [str(n) for n in picked_blue if n is not None and str(n).isdigit()]
    picked_blue = list(dict.fromkeys(picked_blue))

    # 兜底补足（确定性随机）
    while len(picked_red) < front_count:
        n = str(math.floor(rng() * max_red) + 1)
        if n not in picked_red:
            picked_red.append(n)
    while len(picked_blue) < back_count:
        n2 = str(math.floor(rng() * max_blue) + 1)
        if n2 not in picked_blue:
            picked_blue.append(n2)

    picked_red = sorted(int(x) for x in picked_red)
    picked_blue = sorted(int(x) for x in picked_blue)

    if game == "ssq":
        return {"red": picked_red, "blue": picked_blue}
    return {"front": picked_red, "back": picked_blue}


def get_drawn(data, period, game):
    for p in data.get("periods", []):
        if p.get("period") == period:
            if game == "ssq":
                blue = p.get("blue")
                return {"red": p.get("red", []),
                        "blue": [blue] if isinstance(blue, int) else (blue or [])}
            return {"front": p.get("front", []), "back": p.get("back", [])}
    return None


def compute_hits(recommend, drawn, game):
    if game == "ssq":
        return {
            "hit_red": len(set(recommend["red"]) & set(drawn["red"])),
            "hit_blue": len(set(recommend["blue"]) & set(drawn["blue"])),
        }
    return {
        "hit_front": len(set(recommend["front"]) & set(drawn["front"])),
        "hit_back": len(set(recommend["back"]) & set(drawn["back"])),
    }


def main():
    history = load_json(SAGE_PREDICT_FILE, [])
    if not isinstance(history, list):
        history = []

    now_iso = datetime.now(TZ).isoformat()

    for game, model_file, data_file in [
        ("ssq", "ssq_model.json", "ssq_data.json"),
        ("dlt", "dlt_model.json", "dlt_data.json"),
    ]:
        model = load_json(os.path.join(DATA_DIR, model_file))
        data = load_json(os.path.join(DATA_DIR, data_file))
        if not model or not data:
            print(f"[{game.upper()}] 模型或数据缺失，跳过存档")
            continue

        periods = data.get("periods", [])
        if not periods:
            print(f"[{game.upper()}] 无期号数据，跳过")
            continue

        latest_period = periods[0].get("period")  # periods 按最新在前
        target = next_period(latest_period, game)

        def exists_rec(g, sid, per):
            return any(
                r.get("game") == g and r.get("sage_id") == sid
                and r.get("target_period") == per
                for r in history
            )

        # ---- Pass 1：回写所有待补全的历史条目 ----
        for rec in history:
            if rec.get("game") != game or rec.get("drawn") is not None:
                continue
            drawn = get_drawn(data, rec.get("target_period"), game)
            if drawn is not None:
                hits = compute_hits(rec["recommend"], drawn, game)
                rec["drawn"] = drawn
                rec["hit_red"] = hits.get("hit_red")
                rec["hit_blue"] = hits.get("hit_blue")
                rec["hit_front"] = hits.get("hit_front")
                rec["hit_back"] = hits.get("hit_back")
                rec["backfilled_at"] = now_iso
                print(f"[{game.upper()}] 回写: {rec['sage_id']} 目标期 {rec['target_period']} 命中={hits}")
            else:
                print(f"[{game.upper()}] {rec['sage_id']} 目标期 {rec['target_period']} 尚未开奖")

        # ---- Pass 2：为每位人物生成「下一期」实时推荐（去重） ----
        for sage in SAGES:
            if exists_rec(game, sage["id"], target):
                continue
            rec_nums = sage_pick_numbers(sage, game, model, target)
            if not rec_nums:
                continue
            history.append({
                "game": game, "sage_id": sage["id"], "sage_name": sage["name"],
                "strategy": sage["strategy"], "target_period": target,
                "generated_at": now_iso, "model_generated_at": model.get("generated_at"),
                "method": "live",
                "recommend": rec_nums,
                "drawn": None, "hit_red": None, "hit_blue": None,
                "hit_front": None, "hit_back": None,
            })
            print(f"[{game.upper()}] 实时新增: {sage['name']}({sage['strategy']}) 目标期 {target} -> {rec_nums}")

        # ---- Pass 3：回溯模式 —— 用当前模型给最近 N 期已开奖数据生成推荐并立即回测 ----
        # 说明：回溯选号使用「全量模型」（含轻微前视偏差），但命中对照的是该期真实开奖，
        # 因此战绩是真实的，仅用于快速冷启动彩圣榜；实时预测以 method='live' 为准。
        BACKFILL_PERIODS = 50
        drawn_periods = [p.get("period") for p in periods if p.get("period") != target][:BACKFILL_PERIODS]
        for per in drawn_periods:
            for sage in SAGES:
                if exists_rec(game, sage["id"], per):
                    continue
                rec_nums = sage_pick_numbers(sage, game, model, per)
                if not rec_nums:
                    continue
                rec = {
                    "game": game, "sage_id": sage["id"], "sage_name": sage["name"],
                    "strategy": sage["strategy"], "target_period": per,
                    "generated_at": now_iso, "model_generated_at": model.get("generated_at"),
                    "method": "backfill_approx",
                    "recommend": rec_nums,
                }
                drawn = get_drawn(data, per, game)
                if drawn is not None:
                    hits = compute_hits(rec_nums, drawn, game)
                    rec["drawn"] = drawn
                    rec["hit_red"] = hits.get("hit_red")
                    rec["hit_blue"] = hits.get("hit_blue")
                    rec["hit_front"] = hits.get("hit_front")
                    rec["hit_back"] = hits.get("hit_back")
                else:
                    rec["drawn"] = None
                    rec["hit_red"] = rec["hit_blue"] = rec["hit_front"] = rec["hit_back"] = None
                history.append(rec)
        print(f"[{game.upper()}] 回溯完成: 最近 {len(drawn_periods)} 期 × {len(SAGES)} 人物")

    tmp = SAGE_PREDICT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SAGE_PREDICT_FILE)
    print(f"[DONE] 彩圣榜推荐存档已更新: {SAGE_PREDICT_FILE} (共 {len(history)} 条)")


if __name__ == "__main__":
    main()

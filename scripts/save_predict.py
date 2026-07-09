"""
预测存档 + 回写 — P0 Task 1
每次 daily-update workflow 在 model.py 之后运行：
  - 对双色球 / 大乐透，用模型输出的 rf/nb/mc 三组 Top15 评分求和，
    做"确定性"推荐组合（红6蓝1 / 前5后2），追加到 data/predict_history.json
  - 对历史中已存在但尚未回填的条目，用 data 中的实际开奖回写命中情况
设计要点：
  - 组合选取为确定性（无随机），保证同一输入产出一致，便于回测与复现
  - 按 (game, target_period) 去重，避免重复写入同一期
  - 原子写（先写 .tmp 再 os.replace），避免 CI 并发写入损坏文件
依赖：仅标准库（json / os / datetime）。
"""
import json
import os
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TZ = timezone(timedelta(hours=8))

PREDICT_FILE = os.path.join(DATA_DIR, "predict_history.json")


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def next_period(period, game):
    """计算下一期期号。

    SSQ 期号格式：4 位年 + 3 位序号（如 2026075 -> 2026076）
    DLT 期号格式：2 位年 + 3 位序号（如 26076  -> 26077）
    跨年按序号归零处理（如 26153 -> 27001），避免 +1 溢出到错误年份。
    """
    if game == "ssq":
        year = int(period[:4])
        seq = int(period[4:])
        if seq >= 999:
            return f"{year + 1}001"
        return f"{year}{seq + 1:03d}"
    else:  # dlt
        year = int(period[:2])
        seq = int(period[2:])
        if seq >= 999:
            return f"{year + 1:02d}001"
        return f"{year:02d}{seq + 1:03d}"


def combined_pick(group, k):
    """对 rf/nb/mc 三组 Top15 评分求和，取前 k 个（确定性，无随机）。

    group 形如 {"rf": [[ball, score], ...], "nb": [...], "mc": [...]}.
    """
    scores = {}
    for mkey in ("rf", "nb", "mc"):
        for ball, sc in group.get(mkey, []):
            scores[ball] = scores.get(ball, 0.0) + float(sc)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [int(b) for b, _ in ranked[:k]]


def build_recommend(model, game):
    """返回 (recommend_dict, model_top15_dict)。"""
    if game == "ssq":
        red = combined_pick(model["red"], 6)
        blue = combined_pick(model["blue"], 1)
        red.sort(); blue.sort()
        recommend = {"red": red, "blue": blue}
        top15 = {
            "red": {m: model["red"][m] for m in ("rf", "nb", "mc")},
            "blue": {m: model["blue"][m] for m in ("rf", "nb", "mc")},
        }
    else:
        front = combined_pick(model["front"], 5)
        back = combined_pick(model["back"], 2)
        front.sort(); back.sort()
        recommend = {"front": front, "back": back}
        top15 = {
            "front": {m: model["front"][m] for m in ("rf", "nb", "mc")},
            "back": {m: model["back"][m] for m in ("rf", "nb", "mc")},
        }
    return recommend, top15


def get_drawn(data, period, game):
    """从 data 中取某期实际开奖；返回 dict 或 None。"""
    for p in data.get("periods", []):
        if p.get("period") == period:
            if game == "ssq":
                blue = p.get("blue")
                return {
                    "red": p.get("red", []),
                    "blue": [blue] if isinstance(blue, int) else (blue or []),
                }
            else:
                return {
                    "front": p.get("front", []),
                    "back": p.get("back", []),
                }
    return None


def compute_hits(recommend, drawn, game):
    """返回命中字典。SSQ: hit_red / hit_blue；DLT: hit_front / hit_back。"""
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
    history = load_json(PREDICT_FILE, [])
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
                print(f"[{game.upper()}] 回写命中: 目标期 {rec['target_period']}, 命中={hits}")
            else:
                print(f"[{game.upper()}] 目标期 {rec['target_period']} 尚未开奖，暂留空待回写")

        # ---- Pass 2：为"下一期"追加新预测（去重） ----
        exists = any(
            r.get("game") == game and r.get("target_period") == target
            for r in history
        )
        if exists:
            print(f"[{game.upper()}] 目标期 {target} 已存在存档，跳过新增")
            continue

        recommend, top15 = build_recommend(model, game)
        rec = {
            "game": game,
            "target_period": target,
            "generated_at": now_iso,
            "model_generated_at": model.get("generated_at"),
            "recommend": recommend,
            "model_top15": top15,
            "drawn": None,
            "hit_red": None,
            "hit_blue": None,
            "hit_front": None,
            "hit_back": None,
        }
        history.append(rec)
        print(f"[{game.upper()}] 新增预测存档: 目标期 {target}, 推荐={recommend}")

    # 原子写
    tmp = PREDICT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PREDICT_FILE)
    print(f"[DONE] 预测存档已更新: {PREDICT_FILE} (共 {len(history)} 条)")


if __name__ == "__main__":
    main()

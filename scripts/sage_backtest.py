"""
彩圣榜 · 真实回测 — 方案2
读取 data/sage_predict_history.json，对双色球 / 大乐透分别：
  - 按「每位人物 × 每期」的真实命中，累加战绩：
      累计模拟奖金（按真实命中对照奖级表）
      中奖率（命中奖级 > 0 的期数占比）
      平均红球/前区命中数
      最高连红 / 当前连红（连续中奖的期数）
  - 与「理论随机基线」中奖率做 Wilson 区间对比，给出是否显著优于/劣于随机的结论
输出：文本到 stdout + JSON 到 data/sage_backtest_result.json
依赖：仅标准库（json / os / math / random / datetime）。
诚实声明：开奖近似独立随机，本脚本核心价值是「真实量化每个人物的实战表现」，
若全体与随机无显著差异，会如实标注——这正是方案2相比方案1（纯模拟数据）的可信之处。
"""
import json
import math
import os
import random
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TZ = timezone(timedelta(hours=8))

SAGE_PREDICT_FILE = os.path.join(DATA_DIR, "sage_predict_history.json")
RESULT_FILE = os.path.join(DATA_DIR, "sage_backtest_result.json")


# ─── 奖级表（简化固定值，仅用于彩圣榜娱乐性战绩对比）───
def ssq_prize(hr, hb):
    """双色球奖金（元）。hr=红球命中数, hb=蓝球命中数。"""
    if hr == 6 and hb == 1:
        return 5000000          # 一等奖（浮动，取近似）
    if hr == 6 and hb == 0:
        return 100000           # 二等奖
    if hr == 5 and hb == 1:
        return 3000             # 三等奖
    if hr == 5 and hb == 0:
        return 200              # 四等奖
    if hr == 4 and hb == 1:
        return 200              # 四等奖
    if hr == 4 and hb == 0:
        return 10               # 五等奖
    if hr == 3 and hb == 1:
        return 10               # 五等奖
    if hb == 1:
        return 5                # 六等奖（2+1 / 1+1 / 0+1）
    return 0


def dlt_prize(hf, hb):
    """大乐透奖金（元）。hf=前区命中数, hb=后区命中数。"""
    if hf == 5 and hb == 2:
        return 5000000          # 一等奖
    if hf == 5 and hb == 1:
        return 100000           # 二等奖
    if hf == 5 and hb == 0:
        return 10000            # 三等奖
    if hf == 4 and hb == 2:
        return 3000             # 四等奖
    if hf == 4 and hb == 1:
        return 300              # 五等奖
    if hf == 3 and hb == 2:
        return 200             # 六等奖
    if hf == 4 and hb == 0:
        return 100             # 六等奖
    if hf == 2 and hb == 2:
        return 15              # 六等奖
    if hf == 3 and hb == 1:
        return 15              # 六等奖
    if hf == 1 and hb == 2:
        return 15              # 六等奖
    if hf == 0 and hb == 2:
        return 15              # 六等奖
    return 0


def prize_of(game, hit):
    if game == "ssq":
        return ssq_prize(hit.get("hit_red", 0), hit.get("hit_blue", 0))
    return dlt_prize(hit.get("hit_front", 0), hit.get("hit_back", 0))


# ─── 理论随机基线（蒙特卡洛，固定种子可复现）───
def theoretical_win_rate(game, trials=200000, seed=20260720):
    rnd = random.Random(seed)
    if game == "ssq":
        pool, drawn, pick = 33, 6, 6
        blue_pool, blue_drawn, blue_pick = 16, 1, 1
    else:
        pool, drawn, pick = 35, 5, 5
        blue_pool, blue_drawn, blue_pick = 12, 2, 2

    wins = 0
    for _ in range(trials):
        real = set(rnd.sample(range(1, pool + 1), drawn))
        real_blue = set(rnd.sample(range(1, blue_pool + 1), blue_drawn))
        pick_red = set(rnd.sample(range(1, pool + 1), pick))
        pick_blue = set(rnd.sample(range(1, blue_pool + 1), blue_pick))
        hr = len(real & pick_red)
        hb = len(real_blue & pick_blue)
        if prize_of(game, {"hit_red": hr, "hit_blue": hb} if game == "ssq"
                     else {"hit_front": hr, "hit_back": hb}) > 0:
            wins += 1
    return wins / trials


# ─── 统计工具 ────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def verdict_for_rate(obs_k, n, theo_p):
    if n == 0:
        return "样本不足"
    low, high = wilson(obs_k, n)
    if theo_p < low:
        return "显著优于随机"
    if theo_p > high:
        return "显著劣于随机"
    return "与随机无显著差异"


def compute_streaks(records):
    """records: 按 target_period 升序、已开奖的命中列表（win 布尔）。
    返回 (max_streak, current_streak)。"""
    if not records:
        return 0, 0
    max_s = cur = 0
    for r in records:
        if r:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s, cur


def analyze_sage(records, game, theo_win):
    """records: 该人物在该彩种下、已开奖的条目列表（dict）。"""
    n = len(records)
    if n == 0:
        return {"n_evaluated": 0}

    # 按目标期号升序排列，保证连红计算正确
    recs = sorted(records, key=lambda r: str(r.get("target_period")))

    total_prize = 0
    win_count = 0
    hit_red_sum = 0
    hit_blue_sum = 0
    win_flags = []

    for r in recs:
        hit = {
            "hit_red": r.get("hit_red") or 0,
            "hit_blue": r.get("hit_blue") or 0,
            "hit_front": r.get("hit_front") or 0,
            "hit_back": r.get("hit_back") or 0,
        }
        p = prize_of(game, hit)
        total_prize += p
        won = p > 0
        win_flags.append(won)
        if won:
            win_count += 1
        if game == "ssq":
            hit_red_sum += hit["hit_red"]
            hit_blue_sum += hit["hit_blue"]
        else:
            hit_red_sum += hit["hit_front"]
            hit_blue_sum += hit["hit_back"]

    win_rate = win_count / n
    avg_hit_red = hit_red_sum / n
    avg_hit_blue = hit_blue_sum / n if game == "ssq" else 0
    max_streak, current_streak = compute_streaks(win_flags)
    low, high = wilson(win_count, n)

    # 数据构成：区分实时(live)与回溯(backfill_approx)
    live_count = sum(1 for r in recs if r.get("method") == "live")
    backfill_count = n - live_count

    # 诚实判定：回溯含前视偏差，不得宣称「显著优于随机」；
    # 仅当实时样本充足(>=10)时才给真实显著性结论。
    if live_count == 0:
        verdict = "回溯近似·仅供参考"
    elif live_count < 10:
        verdict = "实时样本累积中"
    else:
        verdict = verdict_for_rate(win_count, n, theo_win)

    return {
        "n_evaluated": n,
        "live_count": live_count,
        "backfill_count": backfill_count,
        "total_prize": total_prize,
        "win_rate": round(win_rate, 4),
        "win_rate_ci": [round(low, 4), round(high, 4)],
        "win_count": win_count,
        "avg_hit_red": round(avg_hit_red, 3),
        "avg_hit_blue": round(avg_hit_blue, 3),
        "max_streak": max_streak,
        "current_streak": current_streak,
        "theoretical_win_rate": round(theo_win, 4),
        "verdict": verdict,
    }


def main():
    history = load_json_safe(SAGE_PREDICT_FILE, [])
    if not isinstance(history, list) or not history:
        print("[SAGE BACKTEST] 未找到 sage_predict_history.json，请先运行 sage_predict.py")
        return

    print("=" * 60)
    print("彩圣榜回测引擎 — 开始")
    print(f"时间: {datetime.now(TZ).isoformat()}")
    print("=" * 60)

    report = {
        "generated_at": datetime.now(TZ).isoformat(),
        "games": [],
        "disclaimer": "战绩基于每位人物历史推荐的实测命中，按奖级表折算模拟奖金。"
                      "彩票开奖近似独立随机，任何策略均无法稳定提升中奖概率；"
                      "榜单仅作娱乐性「实战 PK」，样本越少越可能受随机波动影响。",
    }

    for game in ("ssq", "dlt"):
        theo_win = theoretical_win_rate(game)
        recs_game = [r for r in history if r.get("game") == game and r.get("drawn") is not None]
        # 按 sage_id 分组
        by_sage = {}
        for r in recs_game:
            by_sage.setdefault(r.get("sage_id"), []).append(r)

        sages_out = []
        print(f"\n──── {game.upper()} ────")
        print(f"已开奖回测样本（人物×期）: {len(recs_game)}  理论中奖率≈{theo_win:.4f}")

        if not recs_game:
            sages_out = []
        else:
            # 先收集所有人物的静态信息（从任意一条记录取 name/strategy）
            meta = {}
            for r in history:
                if r.get("game") == game:
                    meta[r.get("sage_id")] = {
                        "name": r.get("sage_name"),
                        "strategy": r.get("strategy"),
                    }
            for sage_id, recs in by_sage.items():
                m = meta.get(sage_id, {})
                stat = analyze_sage(recs, game, theo_win)
                stat["id"] = sage_id
                stat["name"] = m.get("name", sage_id)
                stat["strategy"] = m.get("strategy", "")
                sages_out.append(stat)
                print(f"  {stat['name']}({stat['strategy']}): "
                      f"样本{stat['n_evaluated']} 中奖率{stat['win_rate']:.3f} "
                      f"累计奖金{stat['total_prize']} 当前连红{stat['current_streak']} "
                      f"-> {stat['verdict']}")

        sages_out.sort(key=lambda s: s.get("total_prize", 0), reverse=True)
        report["games"].append({
            "game": game,
            "theoretical_win_rate": round(theo_win, 4),
            "n_total_evaluated": len(recs_game),
            "sages": sages_out,
        })

    # ── 一等奖中奖事件收集（用于打开网站自动弹窗）──
    jackpot_events = []
    for r in history:
        if r.get("drawn") is None:
            continue
        game = r.get("game")
        hit = {
            "hit_red": r.get("hit_red") or 0,
            "hit_blue": r.get("hit_blue") or 0,
            "hit_front": r.get("hit_front") or 0,
            "hit_back": r.get("hit_back") or 0,
        }
        is_jackpot = (
            game == "ssq" and hit["hit_red"] == 6 and hit["hit_blue"] == 1
        ) or (
            game == "dlt" and hit["hit_front"] == 5 and hit["hit_back"] == 2
        )
        if is_jackpot:
            jackpot_events.append({
                "game": game,
                "sage_id": r.get("sage_id"),
                "sage_name": r.get("sage_name"),
                "strategy": r.get("strategy"),
                "target_period": r.get("target_period"),
                "prize": prize_of(game, hit),
                "method": r.get("method"),
            })
    report["jackpot_events"] = jackpot_events
    if jackpot_events:
        print(f"\n🎉 检测到一等奖中奖事件 {len(jackpot_events)} 条:")
        for ev in jackpot_events:
            print(f"   {ev['sage_name']}({ev['strategy']}) 第{ev['target_period']}期 {ev['game']} 一等奖 {ev['prize']}元")

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 彩圣榜回测结果已写入: {RESULT_FILE}")


def load_json_safe(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


if __name__ == "__main__":
    main()

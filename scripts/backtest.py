"""
回测引擎 — P0 Task 2
读取 data/predict_history.json，对双色球 / 大乐透分别计算：
  - 综合推荐组合的实测命中率与命中分布
  - 各模型 (rf / nb / mc) Top15 的实测命中率（红球 / 前区）
并与"理论随机基线"做对比（用超几何分布 math.comb 精确计算），
用 Wilson 置信区间判断"是否显著优于 / 劣于随机"。
输出：文本到 stdout + JSON 到 data/backtest_result.json
依赖：仅标准库（json / os / math / datetime）。
设计要点：开奖近似独立随机，本脚本的核心价值是"诚实量化"，
若模型无统计显著优势则如实标注"与随机无显著差异"。
"""
import json
import math
import os
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TZ = timezone(timedelta(hours=8))

PREDICT_FILE = os.path.join(DATA_DIR, "predict_history.json")
RESULT_FILE = os.path.join(DATA_DIR, "backtest_result.json")


# ─── 统计工具 ────────────────────────────────────────
def hypergeom_pmf(N, K, n, k):
    """超几何分布 P(X = k)：从 N 个中抽 n 个，其中 K 个为"成功"，抽到 k 个成功。"""
    if k < max(0, n + K - N) or k > min(n, K):
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def hypergeom_sf(N, K, n, k):
    """P(X >= k)。"""
    return sum(hypergeom_pmf(N, K, n, j) for j in range(k, min(n, K) + 1))


def hypergeom_mean(N, K, n):
    return n * K / N


def wilson(k, n, z=1.96):
    """Wilson 95% 置信区间（对小样本稳健）。返回 (low, high)。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def verdict_for_mean(obs_list, theo_mean):
    """对"均值"类指标用 z 检验判定：比较观测均值与理论基线，
    以样本标准差估计标准误，避免小样本下误报显著。"""
    n = len(obs_list)
    if n < 2:
        return "样本不足"
    mean = sum(obs_list) / n
    var = sum((x - mean) ** 2 for x in obs_list) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return "与随机无显著差异"
    z = (mean - theo_mean) / se
    if z > 1.96:
        return "显著优于随机"
    if z < -1.96:
        return "显著劣于随机"
    return "与随机无显著差异"


def verdict_for_rate(obs_k, n, theo_p):
    if n == 0:
        return "样本不足"
    low, high = wilson(obs_k, n)
    if theo_p < low:
        return "显著优于随机"
    if theo_p > high:
        return "显著劣于随机"
    return "与随机无显著差异"


# ─── 各游戏的指标定义 ────────────────────────────────
# 每组: 名称 / 推荐选取数 pick / 号码池 pool / 实际开出数 drawn / 模型回测用 TopK
GAME_GROUPS = {
    "ssq": [
        {"key": "red", "pick": 6, "pool": 33, "drawn": 6, "topk": 15},
        {"key": "blue", "pick": 1, "pool": 16, "drawn": 1, "topk": 15},
    ],
    "dlt": [
        {"key": "front", "pick": 5, "pool": 35, "drawn": 5, "topk": 15},
        {"key": "back", "pick": 2, "pool": 12, "drawn": 2, "topk": 6},
    ],
}


def analyze_game(history, game):
    recs = [
        r for r in history
        if r.get("game") == game and r.get("drawn") is not None
    ]
    n = len(recs)
    result = {"game": game, "n_evaluated": n, "groups": {}, "note": None}
    if n == 0:
        result["note"] = "暂无已开奖的回测样本（首日运行只有推荐、无命中，次日自动回写）"
        return result

    for g in GAME_GROUPS[game]:
        key = g["key"]
        pick, pool, drawn_n, topk = g["pick"], g["pool"], g["drawn"], g["topk"]

        # 1) 综合推荐组合命中分布
        comb_hits = [r.get(f"hit_{key}") or 0 for r in recs]
        dist = {}
        for h in comb_hits:
            dist[h] = dist.get(h, 0) + 1
        avg_comb = sum(comb_hits) / n
        theo_comb = hypergeom_mean(pool, drawn_n, pick)  # 期望命中数
        # P(至少命中 k) 的理论值（用于对照"至少中1"等）
        theo_rate_ge1 = hypergeom_sf(pool, drawn_n, pick, 1)
        obs_ge1 = sum(1 for h in comb_hits if h >= 1)
        low, high = wilson(obs_ge1, n)

        group_metric = {
            "combination": {
                "avg_hit": round(avg_comb, 4),
                "theoretical_avg_hit": round(theo_comb, 4),
                "hit_distribution": {str(k): v for k, v in sorted(dist.items())},
                "rate_at_least_1": {
                    "observed": round(obs_ge1 / n, 4),
                    "observed_ci": [round(low, 4), round(high, 4)],
                    "theoretical": round(theo_rate_ge1, 4),
                    "verdict": verdict_for_rate(obs_ge1, n, theo_rate_ge1),
                },
                "verdict_avg": verdict_for_mean(comb_hits, theo_comb),
            },
        }

        # 2) 各模型 TopK 命中（红球 / 前区）
        model_metrics = {}
        for m in ("rf", "nb", "mc"):
            model_hits = []
            for r in recs:
                top = r.get("model_top15", {}).get(key, {}).get(m, [])
                top_balls = {int(b) for b, _ in top[:topk]}
                drawn_balls = set(r["drawn"].get(key, []))
                model_hits.append(len(top_balls & drawn_balls))
            if model_hits:
                avg_m = sum(model_hits) / len(model_hits)
                theo_m = hypergeom_mean(pool, drawn_n, topk)
                model_metrics[m] = {
                    "avg_drawn_in_topk": round(avg_m, 4),
                    "theoretical_avg": round(theo_m, 4),
                    "verdict": verdict_for_mean(model_hits, theo_m),
                }
        group_metric["models_topk"] = model_metrics

        result["groups"][key] = group_metric

    return result


def main():
    history = []
    try:
        with open(PREDICT_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("[BACKTEST] 未找到 predict_history.json，请先运行 save_predict.py")
        return

    print("=" * 60)
    print("回测引擎 — 开始")
    print(f"时间: {datetime.now(TZ).isoformat()}")
    print("=" * 60)

    report = {
        "generated_at": datetime.now(TZ).isoformat(),
        "games": [],
        "summary": {},
    }

    for game in ("ssq", "dlt"):
        res = analyze_game(history, game)
        report["games"].append(res)

        print(f"\n──── {game.upper()} ────")
        print(f"已开奖回测样本数: {res['n_evaluated']}")
        if res["note"]:
            print(res["note"])
        for key, gm in res.get("groups", {}).items():
            comb = gm["combination"]
            print(f"\n  [{key}] 综合推荐组合")
            print(f"    平均命中数: {comb['avg_hit']}  (理论随机基线: {comb['theoretical_avg_hit']})")
            print(f"    命中分布: {comb['hit_distribution']}")
            r1 = comb["rate_at_least_1"]
            print(f"    至少中1 实测率: {r1['observed']}  CI95%: {r1['observed_ci']}  理论: {r1['theoretical']}  -> {r1['verdict']}")
            print(f"    平均命中判定: {comb['verdict_avg']}")
            for m, mm in gm.get("models_topk", {}).items():
                print(f"    {m.upper()} Top15 命中均值: {mm['avg_drawn_in_topk']}  (理论: {mm['theoretical_avg']}) -> {mm['verdict']}")

    # 汇总结论
    for res in report["games"]:
        game = res["game"]
        best = []
        for key, gm in res.get("groups", {}).items():
            comb = gm["combination"]
            best.append(
                f"{game}.{key}: 组合{comb['verdict_avg']}; "
                f"至少中1[{comb['rate_at_least_1']['verdict']}]"
            )
        report["summary"][game] = best

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 回测结果已写入: {RESULT_FILE}")


if __name__ == "__main__":
    main()

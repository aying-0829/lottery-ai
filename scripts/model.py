"""
AI 模型评分 — 基于历史开奖数据运行三模型（RF / NB / MC）
输入: data/ssq_data.json, data/dlt_data.json
输出: data/ssq_model.json, data/dlt_model.json
"""
import json
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TZ = timezone(timedelta(hours=8))

# ─── 通用特征工程 ────────────────────────────────────
def build_features(periods, ball_pool, is_front=True):
    """
    构建每期每个号码的特征矩阵。
    特征: 遗漏值, 近5期频率, 近10期频率, 近20期频率, 平均间隔
    """
    pick_count = 6 if is_front else 1
    n = len(periods)
    if n < 20:
        return None, None

    X = []
    y = []
    for i in range(20, n):
        # 提取目标 — 兼容 SSQ(red/blue) 和 DLT(front/back)
        if is_front:
            target_set = set(periods[i].get("red", periods[i].get("front", [])))
        else:
            back_val = periods[i].get("blue", periods[i].get("back", 0))
            target_set = set(back_val if isinstance(back_val, list) else [back_val])

        for ball in ball_pool:
            feats = []

            # 遗漏值：距离上次出现的期数
            skip = 0
            for j in range(i - 1, -1, -1):
                prev_set = set(periods[j].get("red", periods[j].get("front", [])))
                if ball in prev_set:
                    break
                skip += 1
            feats.append(float(skip))

            # 近5/10/20 期频率
            for window in [5, 10, 20]:
                freq = 0
                start = max(0, i - window)
                for j in range(start, i):
                    pset = set(periods[j].get("red", periods[j].get("front", [])))
                    if ball in pset:
                        freq += 1
                feats.append(freq / window)

            # 平均间隔
            appearances = []
            for j in range(i):
                pset = set(periods[j].get("red", periods[j].get("front", [])))
                if ball in pset:
                    appearances.append(j)
            if len(appearances) >= 2:
                gaps = [appearances[k] - appearances[k+1] for k in range(len(appearances)-1)]
                feats.append(sum(gaps) / len(gaps))
            else:
                feats.append(float(n))

            X.append(feats)
            y.append(1 if ball in target_set else 0)

    return np.array(X, dtype=float), np.array(y, dtype=int)


def build_features_dlt_back(periods, ball_pool):
    """大乐透后区特征（选2个，类比蓝球但有2个）"""
    return build_features(periods, ball_pool, is_front=False)


# ─── 蒙特卡洛模拟 ─────────────────────────────────────
def mc_simulation(periods, ball_pool, is_front=True, n_sim=10000, seed=42):
    """蒙特卡洛加权抽样（固定随机种子，保证同输入同输出，便于 CI 增量判断）"""
    pick_count = 6 if is_front else 1
    rng = np.random.default_rng(seed)
    recent = min(50, len(periods))
    recent_periods = periods[:recent]

    freq = Counter()
    for p in recent_periods:
        if is_front or "red" in p:
            balls = p.get("red", p.get("front", []))
        else:
            balls = [p.get("blue", 0)]
        freq.update(balls)

    weights = []
    for ball in ball_pool:
        w = freq.get(ball, 1)  # 出现过的权重大，未出现的至少给1
        weights.append(w)

    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    scores = Counter()
    for _ in range(n_sim):
        chosen = rng.choice(
            ball_pool,
            size=pick_count,
            replace=False,
            p=weights,
        )
        for c in chosen:
            scores[c] += 1

    # 归一化到 0-100
    max_s = max(scores.values()) if scores else 1
    result = []
    for ball in ball_pool:
        score = round(scores.get(ball, 0) / max_s * 100, 2)
        result.append([str(ball), score])

    result.sort(key=lambda x: -x[1])
    return result[:15]


# ─── 模型训练与评分 ──────────────────────────────────
def run_models(periods, ball_pool, is_front=True):
    """训练 RF + NB，返回两个 Top 15 列表"""
    X, y = build_features(periods, ball_pool, is_front)
    if X is None or len(np.unique(y)) < 2:
        # 数据不足，返回空
        empty = [[str(b), 0.0] for b in ball_pool]
        return empty[:15], empty[:15]

    ball_count = len(ball_pool)

    # RF
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    rf_scores = []
    for idx, ball in enumerate(ball_pool):
        # 用该号码的样本均值预测
        idxs = list(range(idx, len(X), ball_count))
        if idxs:
            proba = rf.predict_proba(X[idxs])
            avg = np.mean([p[1] if len(p) > 1 else p[0] for p in proba])
            rf_scores.append([str(ball), round(avg * 100, 2)])
        else:
            rf_scores.append([str(ball), 0.0])

    rf_scores.sort(key=lambda x: -x[1])

    # NB
    nb = GaussianNB()
    nb.fit(X, y)

    nb_scores = []
    for idx, ball in enumerate(ball_pool):
        idxs = list(range(idx, len(X), ball_count))
        if idxs:
            proba = nb.predict_proba(X[idxs])
            avg = np.mean([p[1] if len(p) > 1 else p[0] for p in proba])
            nb_scores.append([str(ball), round(avg * 100, 2)])
        else:
            nb_scores.append([str(ball), 0.0])

    nb_scores.sort(key=lambda x: -x[1])

    return rf_scores[:15], nb_scores[:15]


# ═══════════════════════════════════════════════════════
#  双色球
# ═══════════════════════════════════════════════════════
def save_model_if_changed(out_path, model):
    """仅当评分内容（除 generated_at 外）真正变化时才写文件。

    避免模型确定性后，云端 workflow 仍因时间戳变化每天无意义地
    提交一次数据并触发 Pages 重新部署。
    """
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            changed = False
            for grp in ["red", "blue", "front", "back"]:
                if grp in model and grp in old and model[grp] != old[grp]:
                    changed = True
                    break
            if not changed:
                print(f"[SKIP] 模型内容无变化，跳过写入: {out_path}")
                return
        except Exception:
            pass  # 解析失败则直接重写
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    print(f"[SAVE] 模型已保存: {out_path}")


def process_ssq():
    ssq_file = os.path.join(DATA_DIR, "ssq_data.json")
    if not os.path.exists(ssq_file):
        print("[SSQ] 数据文件不存在，跳过")
        return

    with open(ssq_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    periods = data["periods"]
    print(f"[SSQ] 数据:{len(periods)} 期")

    red_pool = list(range(1, 34))   # 1-33
    blue_pool = list(range(1, 17))   # 1-16

    red_rf, red_nb = run_models(periods, red_pool, is_front=True)
    blue_rf, blue_nb = run_models(periods, blue_pool, is_front=False)

    red_mc = mc_simulation(periods, red_pool, is_front=True)
    blue_mc = mc_simulation(periods, blue_pool, is_front=False)

    model = {
        "#schema": "SSQ_MODEL_V1",
        "generated_at": datetime.now(TZ).isoformat(),
        "data_periods_used": len(periods),
        "red": {"rf": red_rf, "nb": red_nb, "mc": red_mc},
        "blue": {"rf": blue_rf, "nb": blue_nb, "mc": blue_mc},
    }

    out = os.path.join(DATA_DIR, "ssq_model.json")
    save_model_if_changed(out, model)


# ═══════════════════════════════════════════════════════
#  大乐透
# ═══════════════════════════════════════════════════════
def process_dlt():
    dlt_file = os.path.join(DATA_DIR, "dlt_data.json")
    if not os.path.exists(dlt_file):
        print("[DLT] 数据文件不存在，跳过")
        return

    with open(dlt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    periods = data["periods"]
    print(f"[DLT] 数据:{len(periods)} 期")

    front_pool = list(range(1, 36))  # 1-35
    back_pool = list(range(1, 13))    # 1-12

    # 需要适配数据格式 — 字段名不同
    # 把数据统一为 front/back 格式
    if periods and "front" not in periods[0]:
        # 从旧格式转换
        pass  # 当前数据已经是 front/back

    front_rf, front_nb = run_models(periods, front_pool, is_front=True)
    back_rf, back_nb = run_models(periods, back_pool, is_front=False)

    front_mc = mc_simulation(periods, front_pool, is_front=True)
    back_mc = mc_simulation(periods, back_pool, is_front=False)

    model = {
        "#schema": "DLT_MODEL_V1",
        "generated_at": datetime.now(TZ).isoformat(),
        "data_periods_used": len(periods),
        "front": {"rf": front_rf, "nb": front_nb, "mc": front_mc},
        "back": {"rf": back_rf, "nb": back_nb, "mc": back_mc},
    }

    out = os.path.join(DATA_DIR, "dlt_model.json")
    save_model_if_changed(out, model)


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("AI 彩票模型评分 — 开始")
    print(f"时间: {datetime.now(TZ).isoformat()}")
    print("=" * 50)

    process_ssq()
    process_dlt()

    print("\n[DONE] 全部模型已更新")


if __name__ == "__main__":
    main()

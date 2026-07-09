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
import bisect

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TZ = timezone(timedelta(hours=8))

# ─── 通用特征工程 ────────────────────────────────────
def _ball_set(period, is_front):
    """取出某期前/后区号码集合（兼容 SSQ red/blue 与 DLT front/back）。"""
    if is_front:
        return set(period.get("red", period.get("front", [])))
    b = period.get("blue", period.get("back", 0))
    return set(b if isinstance(b, (list, tuple)) else [b])


def build_features(periods, ball_pool, is_front=True, feature_set="v2"):
    """
    构建每期每个号码的特征矩阵。

    feature_set="v1"（旧, 5维）: 遗漏值, 近5/10/20期频率, 平均间隔
    feature_set="v2"（新, 8维）: v1 + 冷热度(近20频率/(遗漏+1)), 遗漏趋势, 号码归一化

    使用预计算出现位置 + bisect 加速，支持全量历史（数千期）高效计算。
    """
    n = len(periods)
    if n < 20:
        return None, None

    # 预计算每个号码的出现期索引（升序），用于 O(log) 查询遗漏与频率
    appear = {b: [] for b in ball_pool}
    for i, p in enumerate(periods):
        for b in _ball_set(p, is_front):
            if b in appear:
                appear[b].append(i)

    max_b = float(ball_pool[-1])
    min_b = float(ball_pool[0])

    X, y = [], []
    for i in range(20, n):
        target_set = _ball_set(periods[i], is_front)
        for ball in ball_pool:
            lst = appear[ball]
            # 最近一次出现位置（< i）
            pos = bisect.bisect_left(lst, i) - 1
            skip = (i - 1 - lst[pos]) if pos >= 0 else float(i)
            # 上一期的遗漏（用于趋势特征）
            if i - 1 >= 0:
                pp = bisect.bisect_left(lst, i - 1) - 1
                skip_prev = (i - 2 - lst[pp]) if pp >= 0 else float(i - 1)
            else:
                skip_prev = 0.0

            # 近 5/10/20 期频率（区间 [start, i) 内的出现次数 / 窗口）
            hi = bisect.bisect_left(lst, i)
            freq5 = (hi - bisect.bisect_left(lst, max(0, i - 5))) / 5.0
            freq10 = (hi - bisect.bisect_left(lst, max(0, i - 10))) / 10.0
            freq20 = (hi - bisect.bisect_left(lst, max(0, i - 20))) / 20.0

            # 平均间隔
            if len(lst) >= 2:
                gaps = [lst[k] - lst[k + 1] for k in range(len(lst) - 1) if lst[k] < i]
                avg_gap = sum(gaps) / len(gaps) if gaps else float(n)
            else:
                avg_gap = float(n)

            feats = [float(skip), freq5, freq10, freq20, avg_gap]
            if feature_set == "v2":
                hot = freq20 / (skip + 1.0)                    # 冷热度：近期热度 / 当前遗漏
                trend = float(skip - skip_prev)                # 遗漏趋势：上升为正，回补为负
                norm = (ball - min_b) / (max_b - min_b + 1e-9)  # 号码大小单调先验
                feats += [hot, trend, norm]

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


# ─── 分类器评分（RF / NB 通用）──────────────────────
def _rank_clf(clf, X, ball_pool, count=15):
    """用训练好的分类器对 ball_pool 中每个号码打分（取该号码样本预测概率均值），返回 Top count。"""
    ball_count = len(ball_pool)
    scores = []
    for idx, ball in enumerate(ball_pool):
        idxs = list(range(idx, len(X), ball_count))
        if idxs:
            proba = clf.predict_proba(X[idxs])
            avg = np.mean([p[1] if len(p) > 1 else p[0] for p in proba])
            scores.append([str(ball), round(avg * 100, 2)])
        else:
            scores.append([str(ball), 0.0])
    scores.sort(key=lambda x: -x[1])
    return scores[:count]


# ─── 确定性共识（三模型评分求和 Top-k）──────────────
def consensus_pick(section, k):
    """对 section 的 rf/nb/mc 三组 Top15 评分求和，取前 k 个（确定性，无随机）。

    返回 dict:
      numbers: 选出的 k 个号码（升序）
      detail:  {str(num): {"in_top15": int, "avg_score": float, "score": float}}
      confidence: 组合整体置信度 = 成员 score 的均值
    score = 0.5*avg_score + 50*(in_top15/3) —— 既看三模型平均评分，也看被几个模型认可
    """
    sums, counts = {}, {}
    for m in ("rf", "nb", "mc"):
        for b, sc in section.get(m, []):
            b = int(b)
            sums[b] = sums.get(b, 0.0) + float(sc)
            counts[b] = counts.get(b, 0) + 1
    ranked = sorted(sums.items(), key=lambda x: -x[1])
    top = [b for b, _ in ranked[:k]]
    detail = {}
    for b in top:
        avg = sums[b] / 3.0
        in15 = counts.get(b, 0)
        score = round(0.5 * avg + 50.0 * (in15 / 3.0), 2)
        detail[str(b)] = {"in_top15": in15, "avg_score": round(avg, 2), "score": score}
    conf = round(sum(d["score"] for d in detail.values()) / k, 2) if detail else 0.0
    return {"numbers": sorted(top), "detail": detail, "confidence": conf}


def _eval_hits(X, y, train, test, ball_pool, is_front, topk, n_est):
    """在 (X,y) 上训练 RF/NB，对 test 评估 Top topk 命中均值，返回 (rf,nb,mc) 均值。"""
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    nb = GaussianNB()
    nb.fit(X, y)

    rf_hits, nb_hits, mc_hits = [], [], []
    for tp in test:
        if is_front:
            target = set(tp.get("red", tp.get("front", [])))
        else:
            b = tp.get("blue", tp.get("back", []))
            target = set(b if isinstance(b, (list, tuple)) else [b])
        rf_top = {int(x) for x, _ in _rank_clf(rf, X, ball_pool, topk)}
        nb_top = {int(x) for x, _ in _rank_clf(nb, X, ball_pool, topk)}
        mc_top = {int(x) for x, _ in mc_simulation(train, ball_pool, is_front, 10000, 42)}
        rf_hits.append(len(rf_top & target))
        nb_hits.append(len(nb_top & target))
        mc_hits.append(len(mc_top & target))
    return float(np.mean(rf_hits)), float(np.mean(nb_hits)), float(np.mean(mc_hits))


def evaluate(periods, ball_pool, is_front=True, holdout=30, n_est=100, compare=True):
    """时序交叉验证（单折 holdout）：用除最后 holdout 期外的数据训练，对留存期评估 Top15 命中均值，
    与理论随机基线对比。compare=True 时额外对比 v1(旧) vs v2(新) 特征集，确认无退化。
    不修改任何产出文件，仅打印日志供 Actions 查看。
    """
    n = len(periods)
    if n < holdout + 20:
        print(f"[EVAL] 数据仅 {n} 期，不足以做 {holdout} 期留一验证，跳过")
        return
    train = periods[:-holdout]
    test = periods[-holdout:]
    label = "红区" if is_front else "蓝/后区"
    topk = min(15, len(ball_pool))
    pick_n = 6 if is_front else (1 if "blue" in test[0] else 2)
    theo = topk * pick_n / len(ball_pool)  # TopK 期望命中数 = topk * 开出数 / 池大小

    X2, y2 = build_features(train, ball_pool, is_front, feature_set="v2")
    if X2 is None or len(np.unique(y2)) < 2:
        print("[EVAL] 训练特征不足，跳过验证")
        return
    avg_rf, avg_nb, avg_mc = _eval_hits(X2, y2, train, test, ball_pool, is_front, topk, n_est)
    print(f"[EVAL] {label} v2(新特征) 验证期 {holdout} 期 (Top{topk}): "
          f"RF={avg_rf:.2f}, NB={avg_nb:.2f}, MC={avg_mc:.2f} | 理论随机基线={theo:.2f}")

    if compare:
        X1, y1 = build_features(train, ball_pool, is_front, feature_set="v1")
        if X1 is not None and len(np.unique(y1)) >= 2:
            a1_rf, a1_nb, a1_mc = _eval_hits(X1, y1, train, test, ball_pool, is_front, topk, n_est)
            print(f"[EVAL] {label} v1(旧特征) 验证期 {holdout} 期 (Top{topk}): "
                  f"RF={a1_rf:.2f}, NB={a1_nb:.2f}, MC={a1_mc:.2f}")
            print(f"[EVAL] {label} 特征对比(v2-v1): RF={avg_rf - a1_rf:+.2f}, NB={avg_nb - a1_nb:+.2f}, "
                  f"MC={avg_mc - a1_mc:+.2f} (差异落在噪声内即视为无退化)")


# ─── 模型训练与评分 ──────────────────────────────────
def run_models(periods, ball_pool, is_front=True):
    """训练 RF + NB，返回两个 Top 15 列表"""
    X, y = build_features(periods, ball_pool, is_front)
    if X is None or len(np.unique(y)) < 2:
        # 数据不足，返回空
        empty = [[str(b), 0.0] for b in ball_pool]
        return empty[:15], empty[:15]

    # RF
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_scores = _rank_clf(rf, X, ball_pool)

    # NB
    nb = GaussianNB()
    nb.fit(X, y)
    nb_scores = _rank_clf(nb, X, ball_pool)

    return rf_scores, nb_scores


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
            for grp in ["red", "blue", "front", "back", "consensus"]:
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

    # 时序交叉验证（P0-T3 + P1 特征对比）：度量模型在留存期的真实命中，供 Actions 日志查看
    evaluate(periods, red_pool, is_front=True, holdout=30, compare=True)
    evaluate(periods, blue_pool, is_front=False, holdout=30, compare=False)

    model = {
        "#schema": "SSQ_MODEL_V1",
        "generated_at": datetime.now(TZ).isoformat(),
        "data_periods_used": len(periods),
        "red": {"rf": red_rf, "nb": red_nb, "mc": red_mc},
        "blue": {"rf": blue_rf, "nb": blue_nb, "mc": blue_mc},
    }

    # 确定性共识（P1-T2）：三模型评分求和取 Top6红 / Top1蓝，带支持度与置信度
    model["consensus"] = {
        "red": consensus_pick(model["red"], 6),
        "blue": consensus_pick(model["blue"], 1),
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

    # 时序交叉验证（P0-T3 + P1 特征对比）
    evaluate(periods, front_pool, is_front=True, holdout=30, compare=True)
    evaluate(periods, back_pool, is_front=False, holdout=30, compare=False)

    model = {
        "#schema": "DLT_MODEL_V1",
        "generated_at": datetime.now(TZ).isoformat(),
        "data_periods_used": len(periods),
        "front": {"rf": front_rf, "nb": front_nb, "mc": front_mc},
        "back": {"rf": back_rf, "nb": back_nb, "mc": back_mc},
    }

    # 确定性共识（P1-T2）：三模型评分求和取 Top5前 / Top2后，带支持度与置信度
    model["consensus"] = {
        "front": consensus_pick(model["front"], 5),
        "back": consensus_pick(model["back"], 2),
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

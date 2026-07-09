"""
AI 模型评分 — 四模型（RF / NB / XGBoost / MC）+ 堆叠融合（Stacking）
输入: data/ssq_data.json, data/dlt_data.json
输出: data/ssq_model.json, data/dlt_model.json

P2 升级（2026-07-10）：
  - 新增 XGBoost 作为第四基准模型（确定性，random_state=42）
  - 新增 Stacking 元学习器：用 RF/NB/XGBoost 三分类器对每号码的「号码级平均概率」
    作元特征，训练 LogisticRegression 融合，输出确定性 Top-k（作为 consensus）
  - 导出 RF 特征重要性，供前端可解释性使用
  - 全部依赖仅 numpy / scikit-learn / xgboost；xgboost 缺失时优雅降级为三模型
"""
import json
import os
import warnings
from datetime import datetime, timezone, timedelta
from collections import Counter
import bisect

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TZ = timezone(timedelta(hours=8))

FEATURE_NAMES = ["skip", "freq5", "freq10", "freq20", "avg_gap", "hot", "trend", "norm"]
RF_N_EST = 200
XGB_N_EST = 200
MODEL_KEYS = ("rf", "nb", "mc", "xgb")  # 共识支持的模型顺序；xgb 缺失时自动跳过


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


# ─── 分类器评分（RF / NB / XGBoost 通用）──────────────
def _rank_clf(clf, X, ball_pool, count=15):
    """用训练好的分类器对 ball_pool 中每个号码打分（取该号码样本预测概率均值），返回 Top count。"""
    ball_count = len(ball_pool)
    scores = []
    for idx, ball in enumerate(ball_pool):
        idxs = list(range(idx, len(X), ball_count))
        if idxs:
            proba = clf.predict_proba(X[idxs])
            avg = np.mean([p[1] if len(p) > 1 else p[0] for p in proba])
            scores.append([str(ball), round(float(avg) * 100, 2)])
        else:
            scores.append([str(ball), 0.0])
    scores.sort(key=lambda x: -x[1])
    return scores[:count]


def _train_base(X, y):
    """训练 RF + NB（+XGBoost 若可用），返回 clfs dict（含 'xgb' 当且仅当 HAS_XGB）。"""
    rf = RandomForestClassifier(n_estimators=RF_N_EST, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    nb = GaussianNB()
    nb.fit(X, y)
    clfs = {"rf": rf, "nb": nb}
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=XGB_N_EST, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        xgb.fit(X, y)
        clfs["xgb"] = xgb
    return clfs


def feature_importance(rf, names=FEATURE_NAMES):
    """RF 特征重要性，按降序返回 [name, imp] 列表。"""
    imp = rf.feature_importances_
    pairs = sorted(
        zip(names, [round(float(x), 4) for x in imp]),
        key=lambda x: -x[1],
    )
    return pairs


def train_and_score(periods, ball_pool, is_front=True):
    """训练基准模型并产出各模型 Top15 评分 + RF 特征重要性。

    返回 (clfs, X, y, rf_scores, nb_scores, xgb_scores, importance)。
    clfs 为 None 表示数据不足（调用方应回退到朴素共识）。
    """
    X, y = build_features(periods, ball_pool, is_front)
    if X is None or len(np.unique(y)) < 2:
        empty = [[str(b), 0.0] for b in ball_pool][:15]
        return None, None, None, empty, empty, empty, []
    clfs = _train_base(X, y)
    rf_scores = _rank_clf(clfs["rf"], X, ball_pool)
    nb_scores = _rank_clf(clfs["nb"], X, ball_pool)
    xgb_scores = (
        _rank_clf(clfs["xgb"], X, ball_pool) if "xgb" in clfs
        else [[str(b), 0.0] for b in ball_pool][:15]
    )
    imp = feature_importance(clfs["rf"])
    return clfs, X, y, rf_scores, nb_scores, xgb_scores, imp


# ─── 确定性共识（多模型评分求和 Top-k）──────────────
def consensus_pick(section, k):
    """对 section 的 rf/nb/mc/xgb 四组 Top15 评分求和，取前 k 个（确定性，无随机）。

    模型集合按实际存在的键动态决定（xgb 缺失时退化为三模型），并记录 model_count。
    返回 dict: numbers / detail / confidence / model_count
    """
    present = [m for m in MODEL_KEYS if section.get(m)]
    sums, counts = {}, {}
    for m in present:
        for b, sc in section[m]:
            b = int(b)
            sums[b] = sums.get(b, 0.0) + float(sc)
            counts[b] = counts.get(b, 0) + 1
    total = len(present) or 1
    ranked = sorted(sums.items(), key=lambda x: -x[1])
    top = [b for b, _ in ranked[:k]]
    detail = {}
    for b in top:
        avg = sums[b] / total
        in15 = counts.get(b, 0)
        score = round(0.5 * avg + 50.0 * (in15 / total), 2)
        detail[str(b)] = {"in_top15": in15, "avg_score": round(avg, 2), "score": score}
    conf = round(sum(d["score"] for d in detail.values()) / k, 2) if detail else 0.0
    return {"numbers": sorted(top), "detail": detail, "confidence": conf, "model_count": total}


# ─── 堆叠融合（Stacking 元学习器）────────────────────
def stacking_pick(clfs, X, y, ball_pool, is_front, k, mc_scores):
    """Stacking：用 RF/NB/XGBoost 三分类器对每号码的「号码级平均概率」作元特征，
    训练 LogisticRegression 元学习器，输出融合后的 Top-k（确定性）。

    - clfs / X / y：来自 train_and_score（与推荐同数据，避免重复训练）
    - mc_scores：MC 的 Top15 列表，仅用于 naive 支持度统计（不进元学习器）
    返回 dict: numbers / detail / confidence / model_count（结构与 consensus_pick 兼容）
    """
    if clfs is None or X is None:
        return consensus_pick(
            {"rf": [], "nb": [], "mc": mc_scores}, k
        )

    rf, nb = clfs["rf"], clfs["nb"]
    xgb = clfs.get("xgb")
    n_ball = len(ball_pool)

    # 每号码的每期概率 → 对号码求均值（元特征，0-1 量纲）
    rf_p = rf.predict_proba(X)[:, 1]
    nb_p = nb.predict_proba(X)[:, 1]
    xgb_p = xgb.predict_proba(X)[:, 1] if xgb is not None else np.zeros(len(X))

    # X 的行按 (period, ball) 排列：ball b 的行索引为 b::n_ball
    rf_avg = np.array([float(np.mean(rf_p[i::n_ball])) for i in range(n_ball)])
    nb_avg = np.array([float(np.mean(nb_p[i::n_ball])) for i in range(n_ball)])
    xgb_avg = np.array([float(np.mean(xgb_p[i::n_ball])) for i in range(n_ball)])

    # 元学习器在「号码级」特征上训练；标签 = 该号码在历史中出现的比例（命中率代理，连续值）
    # 用 Ridge 回归（而非分类器）预测融合后的命中率评分，再按评分排序
    meta_X = np.column_stack([rf_avg, nb_avg, xgb_avg])
    meta_y = y.reshape(n_ball, -1).mean(axis=1)
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, meta_y)
    meta_pred = meta.predict(meta_X)  # 每号码融合评分（≈命中率，0-1）

    # 基础模型 Top15 集合（用于支持度 in_top15 可追溯）
    rf_top15 = set(int(b) for b, _ in _rank_clf(rf, X, ball_pool, 15))
    nb_top15 = set(int(b) for b, _ in _rank_clf(nb, X, ball_pool, 15))
    xgb_top15 = (
        set(int(b) for b, _ in _rank_clf(xgb, X, ball_pool, 15)) if xgb is not None else set()
    )
    mc_top15 = set(int(b) for b, _ in mc_scores)
    present = ["rf", "nb"] + (["xgb"] if xgb is not None else []) + ["mc"]
    top15_map = {"rf": rf_top15, "nb": nb_top15, "xgb": xgb_top15, "mc": mc_top15}
    total = len(present)

    order = sorted(range(n_ball), key=lambda i: -meta_pred[i])
    top_idx = order[:k]
    top = [ball_pool[i] for i in top_idx]
    detail = {}
    for i in top_idx:
        ball = ball_pool[i]
        in15 = sum(ball in top15_map[m] for m in present)
        ms = round(float(np.clip(meta_pred[i], 0, 1)) * 100, 2)
        detail[str(ball)] = {"in_top15": in15, "meta_score": ms, "score": ms}
    conf = (
        round(float(np.mean([detail[str(ball_pool[i])]["score"] for i in top_idx])), 2)
        if top_idx else 0.0
    )
    return {"numbers": sorted(top), "detail": detail, "confidence": conf, "model_count": total}


# ─── 时序交叉验证（stacking / xgb / 多模型对比）──────
def _eval_all(train, test, ball_pool, is_front, topk, feature_set="v2"):
    """在 train 上训练，对 test（留存期）评估各模型 TopK 命中均值。

    返回 dict: {"rf","nb","mc","xgb","naive4","stacking"} 的命中均值（feature_set=v2）。
    feature_set="v1" 时跳过 stacking（stacking 依赖 v2 元特征），仅返回前五项。
    """
    X, y = build_features(train, ball_pool, is_front, feature_set=feature_set)
    if X is None or len(np.unique(y)) < 2:
        return None
    clfs = _train_base(X, y)
    rf, nb = clfs["rf"], clfs["nb"]
    xgb = clfs.get("xgb")

    rf_top = set(int(b) for b, _ in _rank_clf(rf, X, ball_pool, topk))
    nb_top = set(int(b) for b, _ in _rank_clf(nb, X, ball_pool, topk))
    xgb_top = set(int(b) for b, _ in _rank_clf(xgb, X, ball_pool, topk)) if xgb else set()
    mc_top = set(int(b) for b, _ in mc_simulation(train, ball_pool, is_front, 10000, 42))

    # 四模型 naive 共识（评分求和取前 topk）
    naive = {}
    for s in [rf_top, nb_top, mc_top] + ([xgb_top] if xgb else []):
        for ball in s:
            naive[ball] = naive.get(ball, 0) + 1
    naive4_top = set(sorted(naive, key=lambda x: -naive[x])[:topk])

    # stacking（仅 v2）
    stacking_top = None
    if feature_set == "v2":
        stk = stacking_pick(clfs, X, y, ball_pool, is_front, topk,
                            mc_simulation(train, ball_pool, is_front, 10000, 42))
        stacking_top = set(stk["numbers"])

    def _target(tp):
        if is_front:
            return set(tp.get("red", tp.get("front", [])))
        b = tp.get("blue", tp.get("back", []))
        return set(b if isinstance(b, (list, tuple)) else [b])

    rf_h, nb_h, mc_h, xg_h, na_h, st_h = [], [], [], [], [], []
    for tp in test:
        tgt = _target(tp)
        rf_h.append(len(rf_top & tgt))
        nb_h.append(len(nb_top & tgt))
        mc_h.append(len(mc_top & tgt))
        xg_h.append(len(xgb_top & tgt))
        na_h.append(len(naive4_top & tgt))
        if stacking_top is not None:
            st_h.append(len(stacking_top & tgt))

    res = {
        "rf": float(np.mean(rf_h)),
        "nb": float(np.mean(nb_h)),
        "mc": float(np.mean(mc_h)),
        "xgb": float(np.mean(xg_h)),
        "naive4": float(np.mean(na_h)),
    }
    if stacking_top is not None:
        res["stacking"] = float(np.mean(st_h))
    return res


def evaluate(periods, ball_pool, is_front=True, holdout=30, compare=True):
    """时序交叉验证（单折 holdout）：用除最后 holdout 期外的数据训练，对留存期评估
    各模型 Top15 命中均值，与理论随机基线对比。stacking/xgb 一并对比。
    compare=True 时额外对比 v1(旧) vs v2(新) 特征集，确认无退化。
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

    res2 = _eval_all(train, test, ball_pool, is_front, topk, feature_set="v2")
    if res2 is None:
        print("[EVAL] 训练特征不足，跳过验证")
        return
    line = (f"[EVAL] {label} v2 验证期 {holdout} 期 (Top{topk}): "
            f"RF={res2['rf']:.2f}, NB={res2['nb']:.2f}, MC={res2['mc']:.2f}, "
            f"XGB={res2['xgb']:.2f}, naive4={res2['naive4']:.2f}")
    if "stacking" in res2:
        line += f", STACKING={res2['stacking']:.2f}"
    line += f" | 理论随机基线={theo:.2f}"
    print(line)

    if compare:
        res1 = _eval_all(train, test, ball_pool, is_front, topk, feature_set="v1")
        if res1 is not None:
            print(f"[EVAL] {label} v1(旧特征) 验证期 {holdout} 期 (Top{topk}): "
                  f"RF={res1['rf']:.2f}, NB={res1['nb']:.2f}, MC={res1['mc']:.2f}, "
                  f"XGB={res1['xgb']:.2f}, naive4={res1['naive4']:.2f}")
            print(f"[EVAL] {label} 特征对比(v2-v1): "
                  f"RF={res2['rf'] - res1['rf']:+.2f}, NB={res2['nb'] - res1['nb']:+.2f}, "
                  f"MC={res2['mc'] - res1['mc']:+.2f}, XGB={res2['xgb'] - res1['xgb']:+.2f}, "
                  f"naive4={res2['naive4'] - res1['naive4']:+.2f} "
                  f"(差异落在噪声内即视为无退化)")


# ─── 模型训练与评分 ──────────────────────────────────
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
            for grp in ["red", "blue", "front", "back", "consensus", "feature_importance"]:
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
    print(f"[SSQ] 数据:{len(periods)} 期 | XGBoost={'启用' if HAS_XGB else '缺失(降级三模型)'}")

    red_pool = list(range(1, 34))   # 1-33
    blue_pool = list(range(1, 17))   # 1-16

    clfs_r, X_r, y_r, red_rf, red_nb, red_xgb, red_imp = train_and_score(periods, red_pool, is_front=True)
    clfs_b, X_b, y_b, blue_rf, blue_nb, blue_xgb, blue_imp = train_and_score(periods, blue_pool, is_front=False)

    red_mc = mc_simulation(periods, red_pool, is_front=True)
    blue_mc = mc_simulation(periods, blue_pool, is_front=False)

    # 时序交叉验证（度量模型在留存期的真实命中，供 Actions 日志查看）
    evaluate(periods, red_pool, is_front=True, holdout=30, compare=True)
    evaluate(periods, blue_pool, is_front=False, holdout=30, compare=False)

    model = {
        "#schema": "SSQ_MODEL_V2",
        "generated_at": datetime.now(TZ).isoformat(),
        "data_periods_used": len(periods),
        "models": ["rf", "nb", "mc"] + (["xgb"] if HAS_XGB else []),
        "feature_importance": {"red": red_imp, "blue": blue_imp},
        "red": {"rf": red_rf, "nb": red_nb, "mc": red_mc, "xgb": red_xgb},
        "blue": {"rf": blue_rf, "nb": blue_nb, "mc": blue_mc, "xgb": blue_xgb},
    }

    # 确定性共识（P2）：Stacking 融合（四模型）+ 支持度可追溯
    model["consensus"] = {
        "red": stacking_pick(clfs_r, X_r, y_r, red_pool, True, 6, red_mc) if clfs_r
        else consensus_pick({"rf": red_rf, "nb": red_nb, "mc": red_mc}, 6),
        "blue": stacking_pick(clfs_b, X_b, y_b, blue_pool, False, 1, blue_mc) if clfs_b
        else consensus_pick({"rf": blue_rf, "nb": blue_nb, "mc": blue_mc}, 1),
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
    print(f"[DLT] 数据:{len(periods)} 期 | XGBoost={'启用' if HAS_XGB else '缺失(降级三模型)'}")

    front_pool = list(range(1, 36))  # 1-35
    back_pool = list(range(1, 13))    # 1-12

    clfs_f, X_f, y_f, front_rf, front_nb, front_xgb, front_imp = train_and_score(periods, front_pool, is_front=True)
    clfs_k, X_k, y_k, back_rf, back_nb, back_xgb, back_imp = train_and_score(periods, back_pool, is_front=False)

    front_mc = mc_simulation(periods, front_pool, is_front=True)
    back_mc = mc_simulation(periods, back_pool, is_front=False)

    # 时序交叉验证
    evaluate(periods, front_pool, is_front=True, holdout=30, compare=True)
    evaluate(periods, back_pool, is_front=False, holdout=30, compare=False)

    model = {
        "#schema": "DLT_MODEL_V2",
        "generated_at": datetime.now(TZ).isoformat(),
        "data_periods_used": len(periods),
        "models": ["rf", "nb", "mc"] + (["xgb"] if HAS_XGB else []),
        "feature_importance": {"front": front_imp, "back": back_imp},
        "front": {"rf": front_rf, "nb": front_nb, "mc": front_mc, "xgb": front_xgb},
        "back": {"rf": back_rf, "nb": back_nb, "mc": back_mc, "xgb": back_xgb},
    }

    # 确定性共识（P2）：Stacking 融合（四模型）+ 支持度可追溯
    model["consensus"] = {
        "front": stacking_pick(clfs_f, X_f, y_f, front_pool, True, 5, front_mc) if clfs_f
        else consensus_pick({"rf": front_rf, "nb": front_nb, "mc": front_mc}, 5),
        "back": stacking_pick(clfs_k, X_k, y_k, back_pool, False, 2, back_mc) if clfs_k
        else consensus_pick({"rf": back_rf, "nb": back_nb, "mc": back_mc}, 2),
    }

    out = os.path.join(DATA_DIR, "dlt_model.json")
    save_model_if_changed(out, model)


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("AI 彩票模型评分 — 开始 (P2: XGBoost + Stacking)")
    print(f"时间: {datetime.now(TZ).isoformat()}")
    print("=" * 50)

    process_ssq()
    process_dlt()

    print("\n[DONE] 全部模型已更新")


if __name__ == "__main__":
    main()

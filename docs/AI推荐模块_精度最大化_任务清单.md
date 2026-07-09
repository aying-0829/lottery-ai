# AI 推荐模块 · 精度最大化任务清单

> 制定日期：2026-07-09
> 项目定位（用户明确）：**个人使用，非商业营销**。目标 = 在近似随机的物理约束下，把统计方法做到尽可能严谨，并通过回测量化真实命中率。
> 边界说明：开奖近似独立随机，模型命中率理论上趋近随机基线。"精确"= 方法论严谨 + 可量化验证，而非"能预测中奖"。本清单据此排序。

---

## 目标与验收基线

- **理论基线（必须写进回测，作为对照）**：
  - 双色球 6+1：任选 6 红全中概率 = 1/C(33,6) ≈ 1/1,107,568；至少中 1 红 ≈ 1 − C(27,6)/C(33,6) ≈ 66.4%；蓝球命中 = 1/16 = 6.25%。
  - 大乐透 5+2：前区 5 全中 = 1/C(35,5) ≈ 1/324,632；后区 2 全中 = 1/C(12,2) = 1/66。
- **"精确"的合格线**：模型实测命中率与理论基线做显著性对比；若模型无统计显著优势，则如实标注"与随机无异"，这本身就是精确的结论。

---

## P0 — 度量基础（不做这个，精度无从谈起）

### [ ] Task 1: 预测存档（每日追加 + 次日回写）
**Description**: 在 `daily-update.yml` 的 model 步骤后新增一步，把当日 Top-N 推荐（红/蓝、前/后）连同目标期号写入 `data/predict_history.json`（数组追加，由 Actions 提交）。次日运行得知实际开奖后，把命中情况（命中红/蓝数）回写到对应记录。
**Acceptance Criteria**:
- `data/predict_history.json` 每条含：`{target_period, game, recommend:{red:[...],blue:[...]}, drawn:{red:[...],blue:[...]}, hit_red, hit_blue, generated_at}`
- 每日 Actions 运行后该文件被更新并提交；不会重复写同一期
- 历史为空也能初始化（首日只有推荐、无命中，次日补回写）
**Files**: `scripts/save_predict.py`（新增）、`.github/workflows/daily-update.yml`（加步骤）、`data/predict_history.json`（产出）
**Reference**: 计划书:97 `model_results` 规划

### [ ] Task 2: 回测引擎
**Description**: 新增 `scripts/backtest.py`，读取 `predict_history.json`，计算各模型（RF/NB/MC）及"综合推荐"的实测命中率，并与理论基线对比，输出偏差率与简易显著性判断。
**Acceptance Criteria**:
- 输出：推荐组合"至少中 k 红"实测频率 vs 理论频率；Top1 单球命中率 vs 单球理论命中率（6/33）
- 给出"是否显著优于随机"的结论（如二项分布置信区间）
- 纯脚本运行，输出文本/JSON，无外部依赖（仅 numpy）
**Files**: `scripts/backtest.py`（新增）

### [ ] Task 3: 时序交叉验证接入 model.py
**Description**: 将 `run_models` 的训练从"全量训练"改为"时序切分"——用除最后 N 期外的数据训练，对最后 N 期做预测评估，输出各模型在留存期的命中率。
**Acceptance Criteria**:
- `model.py` 新增 `evaluate(periods, ...)` 函数，输出验证期命中率
- 不破坏现有产出（JSON 仍正常生成）
- 验证结果打印到日志，供 Actions 查看
**Files**: `scripts/model.py`

---

## P1 — 特征与建模严谨化

### [ ] Task 4: 特征工程扩充
**Description**: 在 `build_features` 中增补计划书:309-374 已规划但未落地的规则特征：冷热度（hot/cold 标签）、连号概率、和值分布位置、AC 值、区间分布、奇偶比。
**Acceptance Criteria**:
- 新增特征维度在 `build_features` 中可计算且不报 NaN
- 不改变现有 Top15 产出格式（向下兼容）
- 在 `evaluate` 中对比"原特征 vs 新特征"验证期命中率，确认无退化
**Files**: `scripts/model.py`

### [ ] Task 5: 三模型共识合并（收敛到 model.py，去除前端随机）
**Description**: 把"综合推荐"从 `index.html` 的 MC 加权随机采样（refreshRecommend 现只用了 MC），改为 `model.py` 产出的确定性共识：落实计划书:426-427 的"三模型取交集→共识（权重×2），非交集取排名均值权重"。
**Acceptance Criteria**:
- `model.py` 产出 `consensus` 字段（前区 N + 后区 M 的确定组合）
- `index.html` 读取 `consensus` 渲染，不再用 `Math.random()` 生成推荐（消除随机噪声导致的"评分"失真）
- 前端"评分"改为基于共识置信度（如三模型对该组合的支持度），可追溯
**Files**: `scripts/model.py`、`index.html`

### [ ] Task 6: 训练数据回溯全量历史
**Description**: 将 `crawl.py` 抓取范围从近 ~100 期扩展到双色球/大乐透全量历史（数千期），提升统计稳定性。
**Acceptance Criteria**:
- `ssq_data.json` / `dlt_data.json` 期数显著增长（≥ 1000 期）
- 解析逻辑兼容老期号格式，无重复/错期
- 单次抓取不超时（必要时分页）
**Files**: `scripts/crawl.py`

---

## P2 — 模型升级

### [ ] Task 7: 集成学习（XGBoost + Stacking）
**Description**: 在 `model.py` 现有 RF+NB 基础上引入 XGBoost/LightGBM，再做一层元学习器（Stacking）融合三模型。仍在本脚本内、不引入后端。
**Acceptance Criteria**:
- `model.py` 新增 XGBoost 子模型与 Stacking 融合逻辑
- 在 Task 3 的验证框架内对比 Stacking vs 单模型命中率
- 依赖可用（scikit-learn 生态；若需 xgboost 则加入 `requirements.txt`）
**Files**: `scripts/model.py`、`scripts/requirements.txt`

### [ ] Task 8: 真实可解释性（RF 特征重要性 / SHAP）
**Description**: 训练后导出 `rf.feature_importances_` 或 SHAP TreeExplainer 的逐号码贡献，写入 `data/*_model.json`，前端渲染"某号码被推荐的真实依据"。
**Acceptance Criteria**:
- `data/*_model.json` 含每个推荐号码的特征贡献排序
- `index.html` 用真实贡献替换现有写死的套话（index.html:1395）
- 个人使用也便于自查"为什么推这个"
**Files**: `scripts/model.py`、`index.html`

---

## P3 — 进阶实验（可选，验证边际收益）

### [ ] Task 9: 序列模型实验（LSTM / Transformer）
**Description**: 把开奖序列当时间序列，实验 LSTM/Transformer 是否对命中率有边际提升。仅作验证，不为上线承诺。
**Acceptance Criteria**:
- 在 Task 3 验证框架下对比序列模型 vs 基线
- 明确结论："有/无统计显著收益"
- 若收益不显著，记录结论即可，不强制上线
**Files**: `scripts/experiment_seq.py`（新增，独立实验脚本，不影响主流程）

---

## 技术备注
- **架构兼容**：Task 1-8 全部可在现有"静态站点 + GitHub Actions"框架内完成，不破坏云端部署；回测存档走 JSON + Actions 提交。
- **不做**：公开展示合规相关的"改名/免责"改造（个人使用不需要）。
- **预期收益排序**：P0（度量）> P1（严谨）> P2（集成）> P3（前沿实验）。P0 是一切的前提——没有回测，后面所有"更精确"都无法被证明。

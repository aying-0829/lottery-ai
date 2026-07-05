---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 6bc0a8a8fe58de1c28005141d4d9678d_ee85ce11771711f1a7da5254006c9bbf
    ReservedCode1: UbfJrEP1c75S3sFCQO353Ko5sOEn2SRBJidCvTFoBhaAKxCyGXPkXiTLA1PjMm1eo56LeBHZ2mbfiiADJIayy8IidKzjleP0djvdyV3h7JszVpffdINr0nNYrpal+rwYG5d+4hF4vVlhNh/C0rgJWGuZ07LoIbqYnF4cO7fTdSjOo96MntN3AcAvpKM=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 6bc0a8a8fe58de1c28005141d4d9678d_ee85ce11771711f1a7da5254006c9bbf
    ReservedCode2: UbfJrEP1c75S3sFCQO353Ko5sOEn2SRBJidCvTFoBhaAKxCyGXPkXiTLA1PjMm1eo56LeBHZ2mbfiiADJIayy8IidKzjleP0djvdyV3h7JszVpffdINr0nNYrpal+rwYG5d+4hF4vVlhNh/C0rgJWGuZ07LoIbqYnF4cO7fTdSjOo96MntN3AcAvpKM=
---

# AI 彩票分析工具 — API 接口设计规范（数据契约）

> 版本：v1.0 | 日期：2026-07-04 | 基于 api-and-interface-design 方法论

---

## 1. 设计原则

### 1.1 契约优先（Contract First）

本项目采用 **文件系统即接口** 模型——Python 脚本写入 JSON，前端 HTML 通过 `fetch` 读取。数据文件是唯一的集成协议，所有变更必须从 Schema 定义开始。

### 1.2 单版本策略（One-Version Rule）

**不引入版本号字段或 `/v1/` 路径前缀**。原因：

- 本项目是一体仓库，Python 脚本与前端 HTML 始终同步部署（同一 commit）
- GitHub Pages 只托管一个分支（`main`），不存在多版本并行
- 引入版本号只会增加维护负担而无实际收益

### 1.3 Hyrum's Law 防御

"只要有足够多的使用者，API 的所有可观测行为都会被某人依赖。"

对本项目的防御措施：

- **字段含义不可变**：一旦定义，只追加不修改
- **排序不得依赖**：前端不得假设 JSON 数组顺序，必须自行排序
- **数字精度**：评分统一用浮点数，前端用 `toFixed(2)` 格式化

---

## 2. 数据契约 — 4 个 JSON 文件

### 2.1 `data/ssq_data.json` — 双色球历史开奖数据

```jsonc
{
  "#schema": "SSQ_DATA_V1",
  "last_updated": "2026-07-04T02:00:00+08:00",
  "total_periods": 523,
  "periods": [
    {
      "period": "2026071",            // 期号，7位YYYYNNN
      "date": "2026-07-02",
      "red": ["01","06","14","21","27","33"],  // 红球，6个，字符串，升序
      "blue": "09"                    // 蓝球，1个，字符串
    }
  ]
}
```

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `#schema` | string | 固定值 `"SSQ_DATA_V1"` | Schema 标识，用于前端校验 |
| `last_updated` | string(ISO 8601) | 必填 | 数据最后更新时间 |
| `total_periods` | int | ≥0 | 总期数 |
| `periods` | array | 按 period 升序 | 历史数据列表 |
| `periods[].period` | string | 7位数字 | 期号 |
| `periods[].date` | string(YYYY-MM-DD) | 必填 | 开奖日期 |
| `periods[].red` | string[6] | 字符串，值 01-33，无重复，升序 | 红球号码 |
| `periods[].blue` | string | 值 01-16 | 蓝球号码 |

### 2.2 `data/dlt_data.json` — 大乐透历史开奖数据

```jsonc
{
  "#schema": "DLT_DATA_V1",
  "last_updated": "2026-07-04T02:00:00+08:00",
  "total_periods": 498,
  "periods": [
    {
      "period": "2026071",
      "date": "2026-07-02",
      "front": ["05","12","18","24","31"],    // 前区，5个，字符串，升序
      "back": ["03","09"]                     // 后区，2个，字符串，升序
    }
  ]
}
```

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `#schema` | string | 固定值 `"DLT_DATA_V1"` | Schema 标识 |
| `last_updated` | string(ISO 8601) | 必填 | 数据最后更新时间 |
| `total_periods` | int | ≥0 | 总期数 |
| `periods[].front` | string[5] | 值 01-35，无重复，升序 | 前区号码 |
| `periods[].back` | string[2] | 值 01-12，无重复，升序 | 后区号码 |

其余字段同 ssq_data.json。

### 2.3 `data/ssq_model.json` — 双色球模型评分

```jsonc
{
  "#schema": "SSQ_MODEL_V1",
  "generated_at": "2026-07-04T02:00:30+08:00",
  "data_periods_used": 523,
  "red": {
    "rf": [                              // 随机森林 — 红球 Top 15
      ["31", 19.80],
      ["28", 19.27]
    ],
    "nb": [                              // 朴素贝叶斯 — 红球 Top 15
      ["09", 22.38],
      ["31", 21.96]
    ],
    "mc": [                              // 蒙特卡洛 — 红球 Top 15
      ["06", 18.92],
      ["09", 18.71]
    ]
  },
  "blue": {
    "rf":  [["09", 8.52],  ["16", 7.93]],
    "nb":  [["09", 10.21], ["16", 9.85]],
    "mc":  [["09", 9.43],  ["06", 8.91]]
  }
}
```

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `#schema` | string | `"SSQ_MODEL_V1"` | |
| `generated_at` | string(ISO 8601) | 必填 | 模型生成时间 |
| `data_periods_used` | int | ≥0 | 建模使用的总期数 |
| `red.rf` | [string, number][] | 长度=15，评分降序 | RF 红球 Top 15 |
| `red.nb` | [string, number][] | 长度=15，评分降序 | NB 红球 Top 15 |
| `red.mc` | [string, number][] | 长度=15，评分降序 | MC 红球 Top 15 |
| `blue.*` | [string, number][] | 长度=15，评分降序 | 蓝球（同结构） |

### 2.4 `data/dlt_model.json` — 大乐透模型评分

```jsonc
{
  "#schema": "DLT_MODEL_V1",
  "generated_at": "2026-07-04T02:00:30+08:00",
  "data_periods_used": 498,
  "front": {
    "rf":  [["07", 15.18], ["25", 15.05]],
    "nb":  [["33", 16.12], ["19", 16.02]],
    "mc":  [["35", 14.90], ["12", 14.78]]
  },
  "back": {
    "rf":  [["07", 9.21],  ["05", 8.93]],
    "nb":  [["07", 10.44], ["05", 9.87]],
    "mc":  [["07", 9.65],  ["05", 9.12]]
  }
}
```

与 ssq_model.json 结构对称，`red`→`front`，`blue`→`back`。字段约束相同。

---

## 3. 错误处理协议

### 3.1 前端错误状态模型

```
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│   IDLE     │ ───▶ │   LOADING    │ ───▶ │   SUCCESS    │
└────────────┘      └──────┬───────┘      └──────────────┘
                           │
                           ├── HTTP 4xx/5xx ──▶ FAILED (retryable)
                           ├── JSON parse error ─▶ FAILED (retryable)
                           └── Schema 不匹配 ──▶ FAILED (non-retryable)
```

### 3.2 HTTP 状态码语义

| 状态码 | 含义 | 前端行为 |
|--------|------|----------|
| 200 | 成功 | 解析 JSON → 校验 Schema → 渲染 |
| 301/302 | 重定向 | `fetch` 自动跟随（最多 5 跳） |
| 304 | 未修改 | 使用浏览器缓存版本 |
| 403 | 禁止访问 | 显示"数据暂时不可用" |
| 404 | 文件不存在 | 显示"数据文件缺失，请联系维护者" |
| 500+ | 服务端错误 | 显示"服务器异常"，1 次重试，间隔 2 秒 |
| NetworkError | 网络故障 | 显示"网络连接失败"，1 次重试，间隔 2 秒 |

### 3.3 Schema 校验规则

前端在 `JSON.parse` 成功后必须校验 `#schema` 字段：

```javascript
const SCHEMA_MAP = {
  'SSQ_DATA_V1': 'ssq_data.json',
  'DLT_DATA_V1': 'dlt_data.json',
  'SSQ_MODEL_V1': 'ssq_model.json',
  'DLT_MODEL_V1': 'dlt_model.json'
};

function validateSchema(json, expectedSchema) {
  if (json['#schema'] !== expectedSchema) {
    throw new Error(`Schema mismatch: expected ${expectedSchema}, got ${json['#schema']}`);
  }
}
```

Schema 不匹配时抛出 non-retryable 错误（数据格式已变更，重试无意义）。

### 3.4 统一错误响应格式（超出 HTTP 层的情况）

若将来增加 API 层，统一使用以下错误格式：

```json
{
  "error": {
    "code": "DATA_STALE",
    "message": "数据已超过 48 小时未更新",
    "detail": "last_updated: 2026-07-01T02:00:00+08:00",
    "retryable": false
  }
}
```

---

## 4. 向后兼容性保证

### 4.1 修改策略矩阵

| 变更类型 | 允许？ | 条件 |
|----------|--------|------|
| 新增顶层字段 | 是 | 前端做 `?.` 保护，缺失时用默认值 |
| 删除顶层字段 | 否 | 必须同步更新前端 |
| 修改字段类型 | 否 | 破坏性变更，需更新 `#schema` 值 |
| 字段重命名 | 否 | 同上 |
| 调整数组最大长度 | 是（增大） | 前端不 hardcode 长度上限 |
| 评分精度变更 | 是 | 前端统一 `toFixed(2)` 不受影响 |

### 4.2 Schema 演进路径

当必须做出破坏性变更时：

1. 创建新的 `#schema` 值（如 `SSQ_MODEL_V2`）
2. Python 脚本同时写入新 Schema 文件
3. HTML 同时支持新旧 Schema 加载（过渡期）
4. 过渡期后移除旧 Schema 支持

本项目实践中，由于 Python 和 HTML 始终同 commit 部署，破坏性变更可安全进行——只需确保 `#schema` 值更新以触发前端明确失败。

---

## 5. 数据边界验证清单

### 5.1 写入侧（Python 脚本）

| 校验项 | ssq | dlt |
|--------|-----|-----|
| 期号去重（不追加已有期号） | ✓ | ✓ |
| 红球/前区长度 ≥ 1（至少 1 期） | ✓ | ✓ |
| 红球值域 01-33 / 前区 01-35 | ✓ | ✓ |
| 蓝球值域 01-16 / 后区 01-12 | ✓ | ✓ |
| 模型评分非 NaN、非 Infinity | ✓ | ✓ |
| 期号按时间升序 | ✓ | ✓ |
| `last_updated` / `generated_at` 使用 UTC+8 时间 | ✓ | ✓ |

### 5.2 读取侧（HTML 前端）

| 校验项 | 错误处理 |
|--------|----------|
| HTTP 非 200 | 重试逻辑，进入错误状态 |
| JSON 解析失败 | 进入 non-retryable 错误状态 |
| `#schema` 不匹配 | 进入 non-retryable 错误状态 |
| 红球数组长度 ≠ 6 | 跳过该期，记录到 console.warn |
| 号码值域越界 | 跳过该期 |
| `periods` 为空数组 | 显示"暂无数据"占位 |
| 评分数据为空 | 对应模型卡片显示"暂无推荐" |

---

## 6. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-04 | 初始版本，定义 4 个 JSON 的完整 Schema、错误处理协议、兼容性策略 |

---

*（内容由AI生成，仅供参考）*
*（内容由AI生成，仅供参考）*

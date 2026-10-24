# Reward 设计

## 组成

```
score = task_reward + 0.05 * format - LENGTH_PENALTY * max(0, words - SOFT_MAX_TOKENS)
```

`task_reward` 按 `data_source` 路由：

| 分支 | 取值 | 说明 |
|---|---|---|
| MCQ | 0 / 1 | 选项 exact match，支持 CMB 多选 |
| 数值 | 0 / 1 | MedCalc 数值容差 |
| rubric | [-1, 1] | 加权 criterion，负分 criterion 用于惩罚错误断言 |
| safety | [-1, 1] | 优先用 rubric；无 rubric 时用硬编码红旗 |

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `JUDGE_MODEL` | `gpt-4.1-mini` | 迭代期用便宜模型，主表用 `gpt-4.1` |
| `JUDGE_CACHE` | `.cache/judge` | 按 sha256 缓存 verdict，避免重复调用 |
| `RUBRIC_DROPOUT` | `0.0` | 训练时建议 0.2–0.5；同一 `group_id` 共享子集 |
| `LENGTH_PENALTY` | `0.0` | 建议 `0.0005`，配合 `SOFT_MAX_TOKENS=800` |
| `REWARD_MODE` | `train` | 设为 `eval` 时关闭 dropout |
| `RETURN_FLOAT` | 未设置 | 老版本 verl 需要返回 float 时设为 `1` |

## 使用

```bash
python -m rl.rewards.medical_reward   # 本地 smoke（不触发 judge）
```

在 `rl/run_grpo.sh` 里通过 `custom_reward_function.path` / `.name` 接入。

## 反 reward hacking

- **rubric dropout**：group 内共享子 rubric，降低模型对固定评分器的过拟合。
- **负分 criterion**：显式惩罚“无依据的确定诊断”“让患者自行停药”。
- **长度惩罚**：只罚超过软上限的部分，不鼓励无脑变长。
- **监控**：训练日志里同时记录 `task`、`format`、`length_penalty`、平均输出长度、false-claim 率（可每周抽 100 条人工/强 judge 复核）。

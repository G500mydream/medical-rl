# 数据格式

## SFT（LLaMA-Factory）

`data/processed/medical_sft.jsonl`，sharegpt 格式：

```json
{"conversations": [
  {"role": "user", "content": "患者男，65 岁，突发胸痛 2 小时……"},
  {"role": "assistant", "content": "首先评估……"}
]}
```

在 `sft/dataset_info.example.json` 里注册数据集名 `medical_sft`，复制为
`data/processed/dataset_info.json` 后，把 LLaMA-Factory 配置里的 `dataset_dir`
指向 `data/processed`。

建议混合比例（起步）：

| 来源 | 比例 | 作用 |
|---|---|---|
| MedReason 32k / 医学推理链 | 30% | 推理格式 |
| Huatuo-26M / CMtMedQA 中文对话 | 30% | 中文问诊 |
| MedS / PMC-QA 文献问答 | 15% | 证据型回答 |
| HealthBench 风格多轮 + 医生改写 | 15% | 沟通、追问、不确定性 |
| 安全拒答 / 红旗转诊 | 10% | 安全 |

## RL（verl RLHFDataset parquet）

每行一条 prompt：

| 字段 | 类型 | 说明 |
|---|---|---|
| `prompt` | list[dict] | `[{"role": "user", "content": "..."}]` |
| `data_source` | str | 决定走哪个 reward 分支，见下表 |
| `reward_model.ground_truth` | str | MCQ 答案字母 / 数值 / 空 |
| `extra_info.domain_tag` | str | `reasoning` / `communication` / `safety` / `general`，用于 MOPD 路由 |
| `extra_info.rubrics` | list[dict] | `{"criterion": str, "points": float, "type": "positive"\|"negative"}` |
| `extra_info.group_id` | str | 同一 prompt 的多个 rollout 共享，rubric dropout 用同一子集 |

`data_source` 与 reward 分支的对应：

| data_source | reward |
|---|---|
| `medqa` / `medmcqa` / `medxpertqa` / `pubmedqa` / `cmb` / `cmexam` | 选项 exact match |
| `medcalc` / `medcalc_verified` | 数值容差匹配 |
| `healthbench` / `communication` | rubric 打分 |
| `safety` / `medsafetybench` | 安全 rubric（缺失时回退到硬编码规则） |

## 建议的数据构建顺序

1. 先做 **1k 条 HealthBench 子集**（固定 ID、固定 seed），作为迭代期唯一主评测集。
2. RL prompt 池：`2/3` 可验证题（MedQA/MedMCQA/MedCalc）+ `1/3` 开放式（HealthBench/安全）。
3. 每条开放题配 3–8 条 rubric；rubric 来源可以是官方、医生改写，或 GPT-4.1 生成后人工抽检。
4. 保留 `data/samples/` 下的小样本用于 CI/smoke，不要提交真实患者数据。

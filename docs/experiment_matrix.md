# 实验矩阵与评测口径

## 1. 训练 run（全部同一 Qwen3-4B 底座、同一数据预算）

| Run | 训练 | 目的 |
|---|---|---|
| R0 | base | 下界 |
| R1 | +SFT | 数据质量贡献 |
| R2 | +SFT+GRPO | RL 贡献 |
| R3 | +SFT+GRPO+单教师 OPD | 单教师 vs 多教师 |
| **R4** | **+SFT+GRPO+MOPD** | 主方法 |
| R5 | +SFT+GRPO+混合多任务 RL | 算力对齐的强 baseline |
| R6 | +SFT+GRPO+模型合并（TIES/DARE） | 廉价替代 |
| R7 | Qwen3-8B + 同配方 | 效率前沿锚点 |

所有 run 固定：SFT 数据、RL prompt 池、rollout 数（8）、训练步数、max_len、LR schedule。

## 2. 评测矩阵

| 能力轴 | Benchmark | 规模 | 指标 | 频次 |
|---|---|---|---|---|
| 开放式临床对话 | HealthBench（1k 子集 / 全量 / Hard / Consensus） | 1k / 5k / 1k / 3.7k | rubric score | 每 100 步 1k；最终全量 |
| 真实临床工作流 | HealthBench Professional | 525 | 0–1 rubric（长度校正） | 最终 |
| 任务广度 | MedHELM 子集 | 121 tasks | 归一化 win-rate | 最终 |
| 专家推理 | MedXpertQA-Text | 4,460 | accuracy | 每 100 步 |
| 医学知识 | MedQA / MedMCQA / PubMedQA / MedBullets | 1.2k / 4.2k / 1k / ~1k | accuracy | 每 200 步 |
| 计算 | MedCalc-Bench Verified | 1k+ | 数值 exact / 分步 | 每 200 步 |
| 中文 | CMB-Exam / CMB-Clin / CMExam | 11.2k / 74 / 68k | acc / judge | 每 200 步 |
| 安全 | MedSafetyBench / MedHallu | 450 / 10k | safety score / F1 | 最终 |
| 通用保留 | IFEval / MMLU / GSM8K | — | acc | 每 200 步 |
| 效率 | tokens/response、cost/task | — | — | 每次评测 |

## 3. 报告指标

- **绝对分**：mean ± 95% bootstrap CI。
- **相对增益**：`Δ = S_method − S_base`（同底座同数据）。
- **Gap closure**：`(S_method − S_base) / (S_ref − S_base)`，`S_ref` 用同家族更大模型或教师。
- **长度**：原始分、长度校正分、平均 output tokens、每条回答 claim 数。
- **显著性**：paired bootstrap（同一 prompt ID），报告 95% CI 与双侧 p 值。

## 4. 口径红线

1. judge 版本、rubric 版本、max_tokens、temperature 全部写进实验记录；不同版本的数字不放进同一列。
2. HealthBench 必须用 per-criterion 打分，不能用整体 1–10 分代替。
3. MCQ 打乱选项顺序重测一次，报告翻转率。
4. 公开榜（HealthBench/MedQA）分数标注“可能污染”；另设 200–300 条私有 holdout。
5. 外部模型（Baichuan-M2、HuatuoGPT-o1 等）只重跑推理；70B 引论文数字并标注 `reported`。

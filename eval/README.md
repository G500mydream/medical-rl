# 评测

## 工作流

1. 用 vLLM 离线生成回答（`score_rubric.py generate`）；
2. 用 rubric judge 打分（`score_rubric.py score`）；
3. 与 baseline 做 paired bootstrap（`bootstrap_ci.py`）；
4. 通用能力用 `run_general.sh`。

## HealthBench 子集

```bash
python eval/score_rubric.py generate \
  --model outputs/qwen3-4b-mopd \
  --input data/processed/healthbench_1k.jsonl \
  --output outputs/hb_pred.jsonl \
  --tp 2 --max-tokens 1024

python eval/score_rubric.py score \
  --input outputs/hb_pred.jsonl \
  --output outputs/hb_scored.jsonl
```

`*_scored.summary.json` 会给出均值、平均长度和按 theme 的分组均值。

## 与 baseline 比较

```bash
python eval/bootstrap_ci.py outputs/hb_scored.jsonl outputs/baseline_scored.jsonl
```

输出 candidate − baseline 的均值差、95% CI 和双侧 p 值。两条文件必须共享 `id` 字段。

## 其他 benchmark

- MedQA / MedMCQA / PubMedQA / MedXpertQA / CMB / CMExam：用 MedEvalKit 统一跑，导出同一 `id` 口径后复用 `bootstrap_ci.py`。
- 通用能力：`bash eval/run_general.sh <model>`（IFEval / GSM8K / MMLU）。
- 安全：MedSafetyBench 的 450 条对抗 prompt 单独跑，报告 safety score 与 over-refusal 率。

## 口径红线

- 固定 judge 版本（迭代 `gpt-4.1-mini`，主表 `gpt-4.1`），固定 max_tokens，固定 prompt 模板。
- 每次评测保存：模型、解码参数、prompt 文件 hash、judge 模型、rubric 版本。
- 公开榜单分数标注“可能污染”；另设私有 holdout。
- 报告原始分、长度校正分、平均输出长度三者。

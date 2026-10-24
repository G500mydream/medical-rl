# Medical SFT → GRPO → MOPD

医疗健康场景下的三阶段后训练流水线：**SFT 冷启动 → GRPO（rubric + 可验证奖励）→ MOPD（Multi-Teacher On-Policy Distillation）**。

面向 **8× RTX 3090 (24GB)** 单机环境设计：

- 学生：`Qwen/Qwen3-4B-Instruct-2507`，全参数 GRPO；算力允许时换 `Qwen3-8B` + LoRA。
- 教师：3 个 **同底座 LoRA adapter**（医学推理 / 沟通患教 / 安全），共享一份 4B 权重。
- 评测：HealthBench 为主榜，MedXpertQA/MedQA/CMB/MedCalc/MedSafetyBench 为分轴指标。

> 8×3090 上不要尝试 8B 全参 GRPO 或 32B 训练；32B 只做 4-bit 推理评测。

## 目录

```
medical-rl/
├── configs/paths.env.example   # 路径与 judge 配置
├── data/                       # parquet 格式说明 + smoke 数据生成
├── docs/experiment_matrix.md   # R0–R7 实验矩阵与评测口径
├── sft/                        # LLaMA-Factory SFT 配置 + LoRA 合并
├── rl/
│   ├── run_grpo.sh             # verl GRPO 启动脚本（3090 参数）
│   ├── rewards/                # 自定义 reward：rubric / 可验证 / 安全 / 长度
│   └── mopd/                   # 教师池、蒸馏 loss、MOPD 接入说明
├── teachers/                   # T1/T2/T3 教师训练配置
├── eval/                       # vLLM 生成、rubric 打分、bootstrap CI、通用能力
└── scripts/                    # 环境安装、环境自检、流水线入口
```

## 快速开始

```bash
cp configs/paths.env.example configs/paths.env
# 编辑 configs/paths.env，至少设置 OPENAI_API_KEY（judge 用）

bash scripts/setup_env.sh
conda activate medrl
python scripts/check_env.py
python data/make_sample_parquet.py

make sft                 # 阶段 1：SFT（需先准备 data/processed/medical_sft.jsonl）
bash rl/run_grpo.sh      # 阶段 2：GRPO（默认读 data/processed/train.parquet）
bash rl/mopd/run_mopd.sh # 阶段 3：MOPD（需先训好 3 个教师 adapter）
```

评测：

```bash
bash eval/serve_vllm.sh outputs/qwen3-4b-mopd 8000
python eval/score_rubric.py generate --model outputs/qwen3-4b-mopd \
  --input data/processed/healthbench_1k.jsonl --output outputs/hb_pred.jsonl
python eval/score_rubric.py score --input outputs/hb_pred.jsonl \
  --output outputs/hb_scored.jsonl
python eval/bootstrap_ci.py outputs/hb_scored.jsonl outputs/baseline_scored.jsonl
```

## 当前状态

- [x] 目录、配置、启动脚本骨架
- [x] reward 参考实现（rubric / MCQ / 数值 / 安全 / 格式 / 长度惩罚 / 磁盘缓存）
- [x] 教师 adapter 池 + top-k logits 蒸馏 loss 参考实现
- [x] 评测脚本（vLLM 生成、rubric 打分、paired bootstrap、lm-eval）
- [ ] 各数据集到 parquet 的转换脚本（按你的数据源补）
- [ ] verl 内 `mopd` advantage estimator 的正式 patch（接入点见 `rl/mopd/README.md`）

> `rl/rewards/medical_reward.py` 与 `rl/mopd/teacher_pool.py` 是参考实现，接口按 verl 的自定义 reward / 自定义 advantage 约定设计；具体键名以你锁定的 verl release 为准。

## 关键约定

- **数据预算固定**：R0–R7 使用同一份 SFT 数据、同一批 RL prompt、同样的 rollout 数与步数。
- **评测口径固定**：同一批 prompt ID、同一 judge 版本、同一 max_tokens、同一 rubric 版本。
- **外部模型只重跑推理，不重训**；论文报告值与复跑值分栏标注。
- **MOPD 主表必须包含**：单教师 OPD、混合多任务 RL、模型合并三个对照。

## 参考

- verl: https://github.com/volcengine/verl
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- MedEvalKit: https://github.com/alibaba-damo-academy/MedEvalKit
- MOPD-Router: https://github.com/TURLEing/MOPD-Router
- HealthBench (simple-evals): https://github.com/openai/simple-evals

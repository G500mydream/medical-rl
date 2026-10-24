# 教师训练

三个教师都在 **同一 Qwen3-4B 底座** 上训 LoRA，这样 MOPD 阶段只驻留一份底座。
每个教师用与主 GRPO 相同的 rollout / reward 代码，只是数据与奖励权重不同。

## 训练命令

```bash
# 先把 yaml 里的 data 路径转成 parquet，例如 data/processed/t1_train.parquet
TRAIN_FILE=data/processed/t1_train.parquet \
TEACHER_NAME=t1_reasoning \
bash rl/run_grpo.sh

TRAIN_FILE=data/processed/t2_train.parquet \
TEACHER_NAME=t2_communication \
bash rl/run_grpo.sh

TRAIN_FILE=data/processed/t3_train.parquet \
TEACHER_NAME=t3_safety \
bash rl/run_grpo.sh
```

`TEACHER_NAME` 会打开 LoRA（r=32），输出到 `outputs/teachers/<name>`。

## 各教师的数据与奖励

| 教师 | 数据 | 奖励 | 目标能力 |
|---|---|---|---|
| T1 reasoning | MedReason / MedQA / MedXpertQA / MedCalc | 可验证 exact match + 格式 | 知识、推理、计算 |
| T2 communication | HealthBench 训练集 / 患教对话 | rubric（dropout 0.3，负分 criterion） | 追问、共情、不确定性 |
| T3 safety | MedSafetyBench / 拒答 / 红旗转诊 | safety rubric + 负分 | 不伤害、及时转诊 |

## 注意

- 教师不要训太久：300–400 步足够，过训会让教师分布过度尖锐，MOPD 的 token-level 信号反而变差。
- 三个教师共享底座，但**数据不要完全重叠**；否则教师之间没有互补性，MOPD 退化成单教师。
- 训练完确认每个目录下有 `adapter_config.json`；`rl/mopd/run_mopd.sh` 靠它发现教师。
- 如果 verl 导出的 LoRA checkpoint 不是标准 PEFT 目录，用 `peft` 重新保存一次：
  ```python
  from peft import PeftModel
  # 按你的 checkpoint 结构加载后
  model.save_pretrained("outputs/teachers/t1_reasoning")
  ```

# MOPD 接入说明

## 为什么用“同底座 LoRA 教师”

MOPD 的本质是：学生在自己的 rollout 上，逐 token 接收多个教师的监督信号。最贵的部分是教师前向。

- 三个**全参 4B 教师** = 3×8GB 权重 + 三份激活，8×24GB 上很紧。
- 三个**同底座 LoRA adapter** = 1×8GB 底座 + 3×0.2GB adapter，显存几乎不变，只多算几次前向。

所以本项目固定：`Qwen3-4B 底座 + T1/T2/T3 三个 LoRA`。

## 三阶段执行（避免同时驻留）

```
[rollout]     2 GPUs vLLM (TP=2) -> student responses
[score]       切到教师阶段：base + adapter_i 前向，存 top-50 logits
[update]      vLLM 睡眠/释放 -> 8 GPUs ZeRO-3 更新 student
```

这样同一时刻只有“学生+优化器”或“教师+底座”之一在显存里。

## 在 verl 里的三个接入点

1. **优势估计**：在 `core_algos.py` 注册 `adv_estimator=mopd`，把教师信号折算成 token-level advantage；或在 actor loss 里直接加蒸馏项。
2. **教师前向**：在 actor worker 的 `update_actor` 前插入教师 scoring，输入是**当前学生 rollout 的 token**（on-policy 的关键）。
3. **损失**：
   ```python
   loss = policy_loss + lambda_mopd * kl_distill_loss(
       student_logits, teacher_idx, teacher_vals, mask=response_mask,
       temperature=1.0, direction="forward",
   )
   ```
   建议 `lambda_mopd` 从 0.1 起步，按验证集调；同时保留对 SFT 策略的 KL 锚定。

## 路由

- **主实验**：按 `extra_info.domain_tag` 路由到 `reasoning / communication / safety`。
- **消融**：
  - 单教师 OPD（只留 T1）；
  - 均匀聚合（所有教师等权，不看 domain）；
  - label-free token-level router（参考 MOPD-Router 的 ExpertAlign）；
  - DN-MOPD 的域归一化；
  - Δ-MOPD 的 teacher-minus-base 位移。
- **教师可靠性**：学生前缀上的教师置信度可能失真，参考 TrustMOPD 的做法按状态校准。

## 必做的对照

| 对照 | 回答的问题 |
|---|---|
| 单教师 OPD | 多教师是否真的必要 |
| 混合多任务 RL（同算力） | MOPD 是否优于“数据混在一起 RL” |
| 模型合并 TIES/DARE | 蒸馏是否优于权重空间合并 |
| 无 KL 锚定 | 通用能力回退来自哪里 |

## 参考实现

- MOPD-Router: https://github.com/TURLEing/MOPD-Router
- DN-MOPD: https://github.com/LiXin97/DN-MOPD
- NeMo-RL 的 `adv_estimator: opd` / `teacher_model_by_agent_name`

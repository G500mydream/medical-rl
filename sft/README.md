# SFT

## 1. 准备数据

按 `data/README.md` 生成 `data/processed/medical_sft.jsonl`，然后：

```bash
cp sft/dataset_info.example.json data/processed/dataset_info.json
```

## 2. 训练

```bash
llamafactory-cli train sft/qwen3_4b_lora_sft.yaml
# 或
make sft
```

3090 上的默认值：LoRA r=32、batch=1、grad accumulation=8、cutoff 4096、bf16、梯度检查点。
如果 OOM，先把 `cutoff_len` 降到 3072，再把 `per_device_train_batch_size` 保持 1。

## 3. 合并 adapter（RL 阶段用合并后的权重）

```bash
llamafactory-cli export sft/merge_lora.yaml
```

> GRPO 阶段的 actor 直接指向合并后的目录，避免 verl 与 PEFT 的 adapter 加载差异。
> 如果显存紧张，也可以直接在 verl 里用 LoRA，但 4B 全参在 8×3090 上是可行的，优先全参。

## 4. 数据配比建议

见 `data/README.md`。SFT 阶段最重要的是**格式统一**：推理链、最终答案标记、追问句式、拒答模板在训练数据里保持一致，否则 GRPO 的 rubric reward 会把格式噪声当成能力差异。

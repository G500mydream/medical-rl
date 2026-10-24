SHELL := /bin/bash
.PHONY: env check data sft merge grpo mopd serve eval-general clean

MODEL ?= outputs/qwen3-4b-mopd
PORT ?= 8000

env:
	bash scripts/setup_env.sh

check:
	python scripts/check_env.py

data:
	python data/make_sample_parquet.py

sft:
	llamafactory-cli train sft/qwen3_4b_lora_sft.yaml

merge:
	llamafactory-cli export sft/merge_lora.yaml

grpo:
	bash rl/run_grpo.sh

mopd:
	bash rl/mopd/run_mopd.sh

serve:
	bash eval/serve_vllm.sh $(MODEL) $(PORT)

eval-general:
	bash eval/run_general.sh $(MODEL)

clean:
	rm -rf .cache outputs/grpo_smoke

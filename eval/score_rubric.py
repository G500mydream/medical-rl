"""Generate with vLLM and score responses against rubric JSONL.

Input rows need `prompt` (chat messages or string) and `id`; rubric rows also
need `rubrics` or `extra_info.rubrics`.

Examples:
    python eval/score_rubric.py generate --model Qwen/Qwen3-4B-Instruct-2507 \
        --input data/processed/healthbench_1k.jsonl --output outputs/hb_pred.jsonl
    python eval/score_rubric.py score --input outputs/hb_pred.jsonl \
        --output outputs/hb_scored.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_generate(args) -> None:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )

    rows = read_jsonl(args.input)
    conversations = []
    for row in rows:
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        conversations.append(prompt)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.chat(conversations, sampling, use_tqdm=True)

    for row, out in zip(rows, outputs):
        row["response"] = out.outputs[0].text
    write_jsonl(args.output, rows)
    print(f"wrote {args.output} ({len(rows)} rows, max_tokens={args.max_tokens})")


def row_rubrics(row: dict) -> list[dict]:
    rubrics = row.get("rubrics")
    if rubrics is None:
        rubrics = row.get("extra_info", {}).get("rubrics")
    return list(rubrics) if rubrics is not None else []


def cmd_score(args) -> None:
    sys.path.insert(0, str(ROOT))
    from rl.rewards.medical_reward import rubric_score

    rows = read_jsonl(args.input)
    missing_rubrics = 0
    for row in rows:
        rubrics = row_rubrics(row)
        if not rubrics:
            missing_rubrics += 1
        row["score"] = rubric_score(row.get("response", ""), rubrics, dropout=0.0, seed=0)

    write_jsonl(args.output, rows)

    scores = [row["score"] for row in rows]
    lengths = [len(row.get("response", "").split()) for row in rows]
    summary = {
        "n": len(scores),
        "mean": statistics.fmean(scores) if scores else 0.0,
        "mean_words": statistics.fmean(lengths) if lengths else 0.0,
        "min": min(scores) if scores else 0.0,
        "max": max(scores) if scores else 0.0,
        "missing_rubrics": missing_rubrics,
    }

    by_theme: dict[str, list[float]] = {}
    for row in rows:
        theme = row.get("theme") or row.get("extra_info", {}).get("theme")
        if theme:
            by_theme.setdefault(str(theme), []).append(row["score"])
    if by_theme:
        summary["by_theme"] = {k: statistics.fmean(v) for k, v in by_theme.items()}

    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--model", required=True)
    gen.add_argument("--input", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--tp", type=int, default=2)
    gen.add_argument("--gpu-util", type=float, default=0.85)
    gen.add_argument("--max-model-len", type=int, default=4096)
    gen.add_argument("--max-tokens", type=int, default=1024)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--enforce-eager", action="store_true")
    gen.set_defaults(func=cmd_generate)

    score = sub.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

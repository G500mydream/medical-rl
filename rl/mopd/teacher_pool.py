"""Reference multi-teacher pool for MOPD on a shared base model.

All teachers are LoRA adapters on the SAME base checkpoint. That keeps memory
at one 4B copy plus a few hundred MB of adapters, which is what makes MOPD
tractable on 8x 24GB GPUs.

This module is a building block, not a trainer. Wire it into the verl actor
worker:
    1. roll out with the student;
    2. for each teacher, set_adapter() and run topk_logits() over the rollout;
    3. add kl_distill_loss(student_logits, ...) to the policy loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    adapter_path: str
    domain_tags: tuple[str, ...] = ()


class TeacherPool:
    def __init__(
        self,
        base_model: str,
        specs: Sequence[TeacherSpec],
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        if not specs:
            raise ValueError("TeacherPool needs at least one teacher")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(
            base, specs[0].adapter_path, adapter_name=specs[0].name
        )
        for spec in specs[1:]:
            self.model.load_adapter(spec.adapter_path, adapter_name=spec.name)

        self.model.to(device).eval()
        self.specs = {spec.name: spec for spec in specs}
        self._tag_to_teacher = {
            tag: spec.name for spec in specs for tag in spec.domain_tags
        }

    def route(self, domain_tag: str) -> str:
        return self._tag_to_teacher.get(domain_tag, next(iter(self.specs)))

    def set_teacher(self, name: str) -> None:
        if name not in self.specs:
            raise KeyError(f"unknown teacher: {name}")
        self.model.set_adapter(name)

    @torch.no_grad()
    def topk_logits(self, input_ids, attention_mask=None, k: int = 50):
        """Return (topk_indices, topk_logits), shape [B, T, k]."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        values, indices = torch.topk(out, k=k, dim=-1)
        return indices, values

    @torch.no_grad()
    def base_topk_logits(self, input_ids, attention_mask=None, k: int = 50):
        with self.model.disable_adapter():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        values, indices = torch.topk(out, k=k, dim=-1)
        return indices, values


def kl_distill_loss(
    student_logits: torch.Tensor,
    teacher_topk_idx: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    direction: str = "forward",
) -> torch.Tensor:
    """Token-level distillation loss over the teacher's top-k support.

    `teacher_topk_logits` are raw logits, not probabilities. The forward KL is
    approximated on the teacher's top-k support (cheap and stable for 3090s);
    use direction="reverse" for mode-seeking behaviour.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logp_t = torch.log_softmax(teacher_topk_logits / temperature, dim=-1)
    logp_s = torch.log_softmax(student_logits / temperature, dim=-1).gather(
        -1, teacher_topk_idx
    )

    if direction == "forward":
        loss = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    elif direction == "reverse":
        loss = (logp_s.exp() * (logp_s - logp_t)).sum(-1)
    else:
        raise ValueError(f"unknown direction: {direction}")

    if mask is not None:
        mask = mask.to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def build_pool_from_dir(base_model: str, teacher_dir: str, **kwargs) -> TeacherPool:
    """Discover adapters under `teacher_dir/<name>/adapter_config.json`."""
    specs = []
    for path in sorted(Path(teacher_dir).iterdir()):
        if (path / "adapter_config.json").exists():
            specs.append(
                TeacherSpec(
                    name=path.name,
                    adapter_path=str(path),
                    domain_tags=_default_tags(path.name),
                )
            )
    return TeacherPool(base_model, specs, **kwargs)


def _default_tags(name: str) -> tuple[str, ...]:
    name = name.lower()
    if "reason" in name or "knowledge" in name:
        return ("reasoning", "knowledge")
    if "comm" in name or "patient" in name:
        return ("communication",)
    if "safety" in name or "safe" in name:
        return ("safety",)
    return ()

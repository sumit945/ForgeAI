"""Prompt loader. Prompt templates live in forgeai/prompts/*.txt (committed, versioned evidence of how every stage is driven)."""
from __future__ import annotations

import json
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


def render(name: str, **vars) -> tuple[str, str]:
    raw = (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")
    sys_part, user_part = raw.split("### USER", 1)
    system = sys_part.replace("### SYSTEM", "").strip()
    user = user_part.strip()
    for k, v in vars.items():
        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1)
        system = system.replace(f"<<{k}>>", val)
        user = user.replace(f"<<{k}>>", val)
    return system, user

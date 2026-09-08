"""Versioned prompt store + learning-rate promote gate.

Store ported from ltx_research_agent/learn/prompts.py.
Candidates never auto-promote; learning_rate only changes how large a
score gap is required before promote() is advised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

AGENTS = (
    "research",
    "storyboard",
    "image_prompt",
    "motion_prompt",
    "reflection",
    "optimizer",
)


@dataclass
class PromptRecord:
    agent: str
    version: int
    text: str
    path: Path
    current: bool


class PromptStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else Path("prompts")
        self.registry_path = self.root / "prompt_registry.json"

    def registry(self) -> dict[str, dict[str, int]]:
        if not self.registry_path.is_file():
            data = {name: {"current": 1, "previous": 1} for name in AGENTS}
            self._write_registry(data)
            return data
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, data: dict[str, dict[str, int]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def current_version(self, agent: str) -> int:
        return int(self.registry().get(agent, {}).get("current") or 1)

    def path_for(self, agent: str, version: int) -> Path:
        return self.root / agent / f"v{version}.md"

    def load(self, agent: str, version: int | None = None) -> str:
        ver = version or self.current_version(agent)
        path = self.path_for(agent, ver)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prompt {agent} v{ver}: {path}")
        return path.read_text(encoding="utf-8")

    def latest_file_version(self, agent: str) -> int:
        folder = self.root / agent
        if not folder.is_dir():
            return 0
        found = []
        for path in folder.glob("v*.md"):
            try:
                found.append(int(path.stem[1:]))
            except ValueError:
                continue
        return max(found) if found else 0

    def write_candidate(self, agent: str, text: str) -> PromptRecord:
        nxt = self.latest_file_version(agent) + 1
        path = self.path_for(agent, nxt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")
        return PromptRecord(agent, nxt, text, path, current=False)

    def promote(self, agent: str, version: int) -> dict[str, int]:
        data = self.registry()
        prev = int(data.get(agent, {}).get("current") or 1)
        if not self.path_for(agent, version).is_file():
            raise FileNotFoundError(f"Cannot promote missing {agent} v{version}")
        data[agent] = {"current": version, "previous": prev}
        self._write_registry(data)
        return data[agent]


def needed_gain(learning_rate: float) -> float:
    """High rate → small gain required to recommend a promote."""
    rate = max(0.0, min(1.0, float(learning_rate)))
    return round(0.18 * (1.0 - rate), 4)


def should_promote(candidate_score: float, current_score: float, learning_rate: float) -> bool:
    return (float(candidate_score) - float(current_score)) >= needed_gain(learning_rate)

# -*- coding: utf-8 -*-
"""YAML 설정 로더. 모든 기본값은 configs/baseline.yaml 에 있다."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "baseline.yaml"


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, path: str, default=None):
        """점 표기로 꺼낸다. cfg.get('event.threshold_pfu')"""
        node = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def cadence(self) -> int:
        return int(self.get("data.cadence_min", 5))

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))

"""符号资源推理环境。

四个子命令 `generate` / `solve` / `export` / `verify` 共享同一环境内核与轨迹协议。
设计规格见 `doc/symbolic_reasoning_environment_design.md`（1.2 版）。
"""

from .env import (
    SCHEMA_VERSION,
    Issue,
    PlanCheck,
    Rule,
    World,
    WorldError,
    regress,
    step,
    structure_fingerprint,
    verify_plan,
)

__all__ = [
    "SCHEMA_VERSION",
    "Issue",
    "PlanCheck",
    "Rule",
    "World",
    "WorldError",
    "regress",
    "step",
    "structure_fingerprint",
    "verify_plan",
]

"""四种固定 Teacher：正向/反向 × BFS/DFS。

搜索在执行时直接发射事件（第 3.4 节），事件语法见第 4.2 节与附录 B。调度、去重与
终止遵循第 3.3 节的 13 条约定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import env
from .env import SOLVER_VERSION, Issue, Vector, World

MODES = ("forward-bfs", "forward-dfs", "backward-bfs", "backward-dfs")
FORWARD_MODES = ("forward-bfs", "forward-dfs")
BACKWARD_MODES = ("backward-bfs", "backward-dfs")

STATUS_SOLVED = "solved"
STATUS_EXHAUSTED = "exhausted"
STATUS_BUDGET = "budget_exceeded"
STATUS_STATE_LIMIT = "state_limit"
STATUS_ERROR = "error"

DEFAULT_MAX_EXPANSIONS = 4096
DEFAULT_MAX_STATES = 100_000
DEFAULT_MAX_TRACE_LINES = 0  # 0 表示不限

STATUS_TRACE_LIMIT = "trace_limit"


def direction_of(mode: str) -> str:
    _check_mode(mode)
    return "forward" if mode.startswith("forward") else "backward"


def schedule_of(mode: str) -> str:
    _check_mode(mode)
    return "fifo" if mode.endswith("bfs") else "lifo"


def vector_keyword(direction: str) -> str:
    """正向写 STATE，反向写 NEED（第 4.2 节）。"""
    if direction == "forward":
        return "STATE"
    if direction == "backward":
        return "NEED"
    raise ValueError(f"未知方向: {direction}")


def modes_direction_set(modes: Sequence[str]) -> tuple[str, ...]:
    seen = []
    for mode in modes:
        d = direction_of(mode)
        if d not in seen:
            seen.append(d)
    return tuple(seen)


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"未知模式: {mode}；允许 {','.join(MODES)}")


def parse_modes(spec: str) -> tuple[str, ...]:
    """解析 `--modes` 列表，保留给定顺序，拒绝重复。"""
    items = [item.strip() for item in spec.split(",") if item.strip()]
    if not items:
        raise ValueError("--modes 为空")
    for item in items:
        _check_mode(item)
    if len(set(items)) != len(items):
        raise ValueError("--modes 含重复项")
    return tuple(items)


@dataclass(frozen=True)
class Limits:
    max_expansions: int = DEFAULT_MAX_EXPANSIONS
    max_states: int = DEFAULT_MAX_STATES
    max_trace_lines: int = DEFAULT_MAX_TRACE_LINES
    vector_format: str = env.VECTOR_DENSE

    def to_dict(self) -> dict:
        return {
            "max_expansions": self.max_expansions,
            "max_states": self.max_states,
            "max_trace_lines": self.max_trace_lines,
            "vector_format": self.vector_format,
        }


@dataclass
class Cost:
    checked_nodes: int = 0
    generated_nodes: int = 0
    max_frontier: int = 0

    def to_dict(self) -> dict:
        return {
            "checked_nodes": self.checked_nodes,
            "generated_nodes": self.generated_nodes,
            "max_frontier": self.max_frontier,
        }


@dataclass
class SolveResult:
    world_id: str
    mode: str
    status: str
    plan: list[str] | None
    trace: list[str]
    cost: Cost
    limits: Limits
    error: Issue | None = None
    criticality: env.PlanOutcome | None = None

    @property
    def solved(self) -> bool:
        return self.status == STATUS_SOLVED

    def to_dict(self) -> dict:
        return {
            "schema_version": env.SCHEMA_VERSION,
            "world_id": self.world_id,
            "mode": self.mode,
            "solver_version": SOLVER_VERSION,
            "status": self.status,
            "plan": list(self.plan) if self.plan is not None else None,
            "trace": list(self.trace),
            "cost": self.cost.to_dict(),
            "limits": self.limits.to_dict(),
            "error": self.error.to_dict() if self.error else None,
            "criticality": self.criticality.to_dict() if self.criticality else None,
        }

    @staticmethod
    def from_dict(payload: dict) -> "SolveResult":
        cost = payload.get("cost") or {}
        limits = payload.get("limits") or {}
        err = payload.get("error")
        return SolveResult(
            world_id=str(payload["world_id"]),
            mode=str(payload["mode"]),
            status=str(payload["status"]),
            plan=list(payload["plan"]) if payload.get("plan") is not None else None,
            trace=list(payload.get("trace") or []),
            cost=Cost(
                checked_nodes=int(cost.get("checked_nodes", 0)),
                generated_nodes=int(cost.get("generated_nodes", 0)),
                max_frontier=int(cost.get("max_frontier", 0)),
            ),
            limits=Limits(
                max_expansions=int(limits.get("max_expansions", DEFAULT_MAX_EXPANSIONS)),
                max_states=int(limits.get("max_states", DEFAULT_MAX_STATES)),
                max_trace_lines=int(limits.get("max_trace_lines", DEFAULT_MAX_TRACE_LINES)),
                vector_format=str(limits.get("vector_format", env.VECTOR_DENSE)),
            ),
            error=Issue(**err) if err else None,
            criticality=(
                env.PlanOutcome.from_dict(payload["criticality"])
                if payload.get("criticality")
                else None
            ),
        )


def format_vector(
    vec: Sequence[int], symbols: Sequence[str] | None = None, fmt: str = env.VECTOR_DENSE
) -> str:
    return env.format_vector(vec, symbols, fmt)


def solve(world: World, mode: str, limits: Limits | None = None) -> SolveResult:
    """运行一个固定 Teacher，返回最终方案与完整搜索事件。

    首解即停。预算在取出节点之前判断（约定 11），因此 `budget_exceeded` 的事件前缀
    末条为 CLOSE，不含残缺的 EXPAND。
    """
    _check_mode(mode)
    limits = limits or Limits()
    direction = direction_of(mode)
    fifo = schedule_of(mode) == "fifo"
    keyword = vector_keyword(direction)
    forward = direction == "forward"
    fmt = limits.vector_format
    if fmt not in env.VECTOR_FORMATS:
        raise ValueError(f"未知向量编码: {fmt}")
    line_cap = limits.max_trace_lines or None

    def render(vec: Vector) -> str:
        return env.format_vector(vec, world.symbols, fmt)

    root: Vector = world.initial if forward else world.goal
    vectors: list[Vector] = [root]
    parent: list[int] = [-1]
    via: list[str | None] = [None]
    visited: dict[Vector, int] = {root: 0}
    frontier: list[int] = [0]

    trace: list[str] = []
    checked = 0
    max_frontier = 1
    status: str | None = None
    plan: list[str] | None = None
    error: Issue | None = None

    while frontier:
        if checked >= limits.max_expansions:
            status = STATUS_BUDGET
            break
        if line_cap is not None and len(trace) >= line_cap:
            status = STATUS_TRACE_LIMIT
            break
        idx = frontier.pop(0) if fifo else frontier.pop()
        checked += 1
        vec = vectors[idx]
        trace.append(f"EXPAND n{idx} {keyword} {render(vec)}")

        if _terminal(world, vec, forward):
            trace.append(f"GOAL n{idx}")
            plan = _restore_plan(parent, via, idx, forward)
            check = env.verify_plan(world, plan)
            if not check.valid:
                # Teacher 恢复的方案未通过独立执行，属于实现异常。
                status = STATUS_ERROR
                error = check.error
                break
            trace.append("PLAN" + ("" if not plan else " " + " ".join(plan)))
            trace.append("END")
            status = STATUS_SOLVED
            break

        new_nodes: list[int] = []
        hit_state_limit = False
        for rule in world.rules:
            child = (
                env.step(vec, rule)
                if forward
                else env.regress(vec, rule, world.total_units)
            )
            if child is None:
                continue
            known = visited.get(child)
            if known is not None:
                trace.append(f"DUP n{known} VIA {rule.id}")
                continue
            if len(visited) >= limits.max_states:
                hit_state_limit = True
                break
            new_idx = len(vectors)
            vectors.append(child)
            parent.append(idx)
            via.append(rule.id)
            visited[child] = new_idx
            new_nodes.append(new_idx)
            trace.append(f"NEW n{new_idx} VIA {rule.id} {keyword} {render(child)}")

        if hit_state_limit:
            status = STATUS_STATE_LIMIT
            break

        trace.append(f"CLOSE n{idx}")
        frontier.extend(new_nodes if fifo else reversed(new_nodes))
        max_frontier = max(max_frontier, len(frontier))

    if status is None:
        status = STATUS_EXHAUSTED

    cost = Cost(checked_nodes=checked, generated_nodes=len(vectors), max_frontier=max_frontier)
    return SolveResult(
        world_id=world.world_id,
        mode=mode,
        status=status,
        plan=plan if status == STATUS_SOLVED else None,
        trace=trace,
        cost=cost,
        limits=limits,
        error=error,
    )


def _terminal(world: World, vec: Vector, forward: bool) -> bool:
    if forward:
        return env.meets_goal(vec, world.goal)
    return env.demand_met(vec, world.initial)


def _restore_plan(
    parent: Sequence[int], via: Sequence[str | None], idx: int, forward: bool
) -> list[str]:
    """沿 parent 链恢复正向可执行方案。

    自节点向根走得到 [r_L, ..., r_1]。正向需要根到节点的顺序，故反转；反向的正向方案
    恰为该反转序列本身（第 3.2 节）。
    """
    chain: list[str] = []
    cur = idx
    while parent[cur] != -1:
        rule_id = via[cur]
        assert rule_id is not None
        chain.append(rule_id)
        cur = parent[cur]
    return list(reversed(chain)) if forward else chain


def cost_within_budget(result: SolveResult, budget: int) -> bool | None:
    """第 8.3 节的预算内判定表。None 表示 unknown。"""
    if result.status == STATUS_SOLVED:
        return result.cost.checked_nodes <= budget
    if result.status == STATUS_BUDGET:
        return False if result.limits.max_expansions >= budget else None
    if result.status == STATUS_EXHAUSTED:
        return False
    return None


# --------------------------------------------------------------------------- #
# 分歧位置（第 5.1 节）
# --------------------------------------------------------------------------- #


def common_prefix_length(left: Sequence, right: Sequence) -> int:
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n


def count_expansions(lines: Sequence[str]) -> int:
    return sum(1 for line in lines if line.startswith("EXPAND "))


def mode_pairs(modes: Sequence[str]) -> list[tuple[str, str]]:
    """按 --modes 的给定顺序取全部两元组合。"""
    out: list[tuple[str, str]] = []
    for i, left in enumerate(modes):
        for right in modes[i + 1 :]:
            out.append((left, right))
    return out


def divergence_record(
    world_id: str, pair: tuple[str, str], left: SolveResult, right: SolveResult
) -> dict:
    """逐题逐模式对的分歧结构量，契约见第 9.4 节。"""
    both = left.solved and right.solved
    shared_lines = common_prefix_length(left.trace, right.trace) if both else None
    payload = {
        "schema_version": env.SCHEMA_VERSION,
        "world_id": world_id,
        "pair": [pair[0], pair[1]],
        "both_solved": both,
        "shared_lines": shared_lines,
        "shared_expansions": (
            count_expansions(left.trace[:shared_lines]) if both else None
        ),
        "plan_shared_prefix": (
            common_prefix_length(left.plan or [], right.plan or []) if both else None
        ),
        "trace_identical": (left.trace == right.trace) if both else None,
        "plan_identical": (left.plan == right.plan) if both else None,
        "trace_lines": [len(left.trace), len(right.trace)],
        "plan_lengths": [
            len(left.plan) if left.plan is not None else None,
            len(right.plan) if right.plan is not None else None,
        ],
        "checked_nodes": [left.cost.checked_nodes, right.cost.checked_nodes],
    }
    return payload


def frontier_trace(world: World, mode: str, limits: Limits | None = None) -> list[int]:
    """重放该模式，记录每次 EXPAND 之前的 frontier 规模。"""
    limits = limits or Limits()
    forward = direction_of(mode) == "forward"
    fifo = schedule_of(mode) == "fifo"
    root: Vector = world.initial if forward else world.goal
    vectors = [root]
    visited = {root: 0}
    frontier = [0]
    sizes: list[int] = []
    while frontier and len(sizes) < limits.max_expansions:
        sizes.append(len(frontier))
        idx = frontier.pop(0) if fifo else frontier.pop()
        vec = vectors[idx]
        if _terminal(world, vec, forward):
            break
        fresh: list[int] = []
        for rule in world.rules:
            child = (
                env.step(vec, rule)
                if forward
                else env.regress(vec, rule, world.total_units)
            )
            if child is None or child in visited:
                continue
            visited[child] = len(vectors)
            vectors.append(child)
            fresh.append(visited[child])
        frontier.extend(fresh if fifo else reversed(fresh))
    return sizes

"""符号资源推理环境的核心语义。

包含世界定义与校验、正向状态转移、反向需求回归、最终方案验证，以及附录 C 的错误码
与附录 D 的标识哈希。本模块不依赖项目内其他模块。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

SCHEMA_VERSION = "1.5"
GENERATOR_VERSION = "1.0"
SOLVER_VERSION = "1.0"
EXPORTER_VERSION = "1.0"

Vector = tuple[int, ...]

RULE_ID_RE = re.compile(r"^r(?:0|[1-9][0-9]*)$")
NODE_ID_RE = re.compile(r"^n(?:0|[1-9][0-9]*)$")
INT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")


# --------------------------------------------------------------------------- #
# 错误码（附录 C）
# --------------------------------------------------------------------------- #

PARSE_ERROR = "PARSE_ERROR"
VECTOR_LENGTH = "VECTOR_LENGTH"
INCOMPLETE_TRACE = "INCOMPLETE_TRACE"
DIRECTION_MISMATCH = "DIRECTION_MISMATCH"
ROOT_MISMATCH = "ROOT_MISMATCH"
UNKNOWN_RULE = "UNKNOWN_RULE"
UNKNOWN_NODE = "UNKNOWN_NODE"
DUPLICATE_NODE_ID = "DUPLICATE_NODE_ID"
BAD_NODE_ORDER = "BAD_NODE_ORDER"
WRONG_STATE = "WRONG_STATE"
MISSING_SUCCESSOR = "MISSING_SUCCESSOR"
EXTRA_SUCCESSOR = "EXTRA_SUCCESSOR"
SUCCESSOR_ORDER = "SUCCESSOR_ORDER"
DUP_TARGET_MISMATCH = "DUP_TARGET_MISMATCH"
MISSING_CLOSE = "MISSING_CLOSE"
UNEXPECTED_EVENT = "UNEXPECTED_EVENT"
BAD_FRONTIER_CHOICE = "BAD_FRONTIER_CHOICE"
BAD_GOAL_CLAIM = "BAD_GOAL_CLAIM"
INSUFFICIENT_RESOURCE = "INSUFFICIENT_RESOURCE"
GOAL_NOT_MET = "GOAL_NOT_MET"
PLAN_TRACE_MISMATCH = "PLAN_TRACE_MISMATCH"
PLAN_TOO_LONG = "PLAN_TOO_LONG"
TRACE_TOO_LONG = "TRACE_TOO_LONG"
VECTOR_FORMAT_MIXED = "VECTOR_FORMAT_MIXED"
VECTOR_SYMBOL_UNKNOWN = "VECTOR_SYMBOL_UNKNOWN"
VECTOR_SYMBOL_ORDER = "VECTOR_SYMBOL_ORDER"
VECTOR_SYMBOL_DUPLICATE = "VECTOR_SYMBOL_DUPLICATE"

ALL_ERROR_CODES = (
    PARSE_ERROR,
    VECTOR_LENGTH,
    INCOMPLETE_TRACE,
    DIRECTION_MISMATCH,
    ROOT_MISMATCH,
    UNKNOWN_RULE,
    UNKNOWN_NODE,
    DUPLICATE_NODE_ID,
    BAD_NODE_ORDER,
    WRONG_STATE,
    MISSING_SUCCESSOR,
    EXTRA_SUCCESSOR,
    SUCCESSOR_ORDER,
    DUP_TARGET_MISMATCH,
    MISSING_CLOSE,
    UNEXPECTED_EVENT,
    BAD_FRONTIER_CHOICE,
    BAD_GOAL_CLAIM,
    INSUFFICIENT_RESOURCE,
    GOAL_NOT_MET,
    PLAN_TRACE_MISMATCH,
    PLAN_TOO_LONG,
    TRACE_TOO_LONG,
    VECTOR_FORMAT_MIXED,
    VECTOR_SYMBOL_UNKNOWN,
    VECTOR_SYMBOL_ORDER,
    VECTOR_SYMBOL_DUPLICATE,
)

# Human-readable report labels; validation continues to use the stable codes.
ERROR_CODE_LABELS = {
    PARSE_ERROR: "事件语法错误",
    VECTOR_LENGTH: "资源向量长度错误",
    INCOMPLETE_TRACE: "回答不完整",
    DIRECTION_MISMATCH: "搜索方向不一致",
    ROOT_MISMATCH: "根节点状态错误",
    UNKNOWN_RULE: "引用不存在的规则",
    UNKNOWN_NODE: "引用未登记的节点",
    DUPLICATE_NODE_ID: "节点编号重复",
    BAD_NODE_ORDER: "节点编号顺序错误",
    WRONG_STATE: "资源状态计算错误",
    MISSING_SUCCESSOR: "遗漏后继事件",
    EXTRA_SUCCESSOR: "输出多余或类型错误的后继事件",
    SUCCESSOR_ORDER: "后继规则顺序错误",
    DUP_TARGET_MISMATCH: "重复状态引用错误",
    MISSING_CLOSE: "缺少 CLOSE 事件",
    UNEXPECTED_EVENT: "事件位置或节点引用错误",
    BAD_FRONTIER_CHOICE: "展开节点选择错误",
    BAD_GOAL_CLAIM: "错误声明达到目标",
    INSUFFICIENT_RESOURCE: "最终方案执行时资源不足",
    GOAL_NOT_MET: "最终方案未达到目标",
    PLAN_TRACE_MISMATCH: "最终方案与搜索路径不一致",
    PLAN_TOO_LONG: "方案步数超过验证上限",
    TRACE_TOO_LONG: "事件数超过验证上限",
    VECTOR_FORMAT_MIXED: "资源向量格式混用",
    VECTOR_SYMBOL_UNKNOWN: "引用不存在的资源名",
    VECTOR_SYMBOL_ORDER: "资源名顺序错误",
    VECTOR_SYMBOL_DUPLICATE: "资源名重复",
}

ERROR_CODE_DESCRIPTIONS = {
    PARSE_ERROR: "事件行不符合协议语法，例如字段缺失、多余字段、非法编号或空格格式错误。",
    VECTOR_LENGTH: "稠密向量的计数项数与题目中的资源种类数不一致。",
    INCOMPLETE_TRACE: "回答不完整，例如缺少 `</think>`、PLAN 或 END；是否因长度上限截断需结合 finish_reason 判断。",
    DIRECTION_MISMATCH: "混用了正向 STATE 与反向 NEED，或轨迹方向与指定搜索模式不符。",
    ROOT_MISMATCH: "根节点内容错误：正向应等于 INITIAL，反向应等于 GOAL。",
    UNKNOWN_RULE: "引用了题目 RULES 中不存在的规则编号。",
    UNKNOWN_NODE: "引用了尚未登记的节点编号。",
    DUPLICATE_NODE_ID: "NEW 重复使用了已经登记的节点编号。",
    BAD_NODE_ORDER: "新节点编号未按发现顺序递增，例如应创建 n3 却创建 n5。",
    WRONG_STATE: "EXPAND 或 NEW 写出的资源数量与规则重放结果不一致，例如资源少扣、多扣或无故增加。",
    MISSING_SUCCESSOR: "当前节点尚有应输出的 NEW 或 DUP，模型便提前输出了 CLOSE。",
    EXTRA_SUCCESSOR: "输出了不应出现的后继事件，例如规则条件不满足、重复输出已处理的规则，或应为 DUP 却写成 NEW。",
    SUCCESSOR_ORDER: "后继没有按规则编号的数值升序输出，例如应先处理 r7，却先输出 r13 的后继。",
    DUP_TARGET_MISMATCH: "DUP 指向了错误的已有节点，或该结果应创建 NEW，模型却使用了 DUP。",
    MISSING_CLOSE: "非终止节点的展开缺少 CLOSE，未按协议结束该节点的后继枚举。",
    UNEXPECTED_EVENT: "事件出现在当前阶段不允许的位置，或引用的节点与当前事件不符，例如 GOAL 后继续输出 NEW。",
    BAD_FRONTIER_CHOICE: "EXPAND 选择的节点不在待展开集合中；模式验证还检查是否符合 BFS 或 DFS 的取节点顺序。",
    BAD_GOAL_CLAIM: "模型声明 GOAL，但该节点不满足对应搜索方向的终止条件。",
    INSUFFICIENT_RESOURCE: "执行最终 PLAN 时，某一步缺少该规则要求消耗的资源。",
    GOAL_NOT_MET: "最终 PLAN 可以执行，但执行结束后的资源状态未满足 GOAL。",
    PLAN_TRACE_MISMATCH: "最终 PLAN 与沿 GOAL 节点父节点链恢复出的规则序列不一致。",
    PLAN_TOO_LONG: "最终方案的规则步数超过 --max-plan-steps。",
    TRACE_TOO_LONG: "回答的事件行数超过 --max-events。",
    VECTOR_FORMAT_MIXED: "同一轨迹混用了稠密计数向量与稀疏的资源名:数量格式。",
    VECTOR_SYMBOL_UNKNOWN: "稀疏向量包含 RESOURCES 中不存在的资源名。",
    VECTOR_SYMBOL_ORDER: "稀疏向量中的资源名未按 RESOURCES 的顺序排列；这是格式约束，资源数量可能仍然正确。",
    VECTOR_SYMBOL_DUPLICATE: "同一个稀疏向量中重复出现了相同资源名，例如 A:1 A:2。",
}


class WorldError(ValueError):
    """世界定义不满足首版约束。入口报错，不做静默修正。"""


@dataclass(frozen=True)
class Issue:
    """单个违例：稳定错误码加位置。"""

    code: str
    line: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "line": self.line, "detail": self.detail}


# --------------------------------------------------------------------------- #
# 世界
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rule:
    id: str
    consume: Vector
    produce: Vector

    @property
    def index(self) -> int:
        return int(self.id[1:])

    def to_dict(self) -> dict:
        return {"id": self.id, "consume": list(self.consume), "produce": list(self.produce)}


@dataclass(frozen=True)
class World:
    symbols: tuple[str, ...]
    initial: Vector
    goal: Vector
    rules: tuple[Rule, ...]
    world_id: str = ""
    group_id: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_resources(self) -> int:
        return len(self.symbols)

    @property
    def total_units(self) -> int:
        """初态总资源量 m。正向转移不增加总量，故也是可达状态与反向需求的上限。"""
        return sum(self.initial)

    def rule(self, rule_id: str) -> Rule | None:
        return self._index.get(rule_id)

    @property
    def _index(self) -> dict[str, Rule]:
        cached = self.__dict__.get("_rule_index")
        if cached is None:
            cached = {r.id: r for r in self.rules}
            object.__setattr__(self, "_rule_index", cached)
        return cached

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "world_id": self.world_id,
            "group_id": self.group_id,
            "symbols": list(self.symbols),
            "initial": list(self.initial),
            "goal": list(self.goal),
            "rules": [r.to_dict() for r in self.rules],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(payload: dict) -> "World":
        try:
            symbols = tuple(payload["symbols"])
            initial = tuple(int(x) for x in payload["initial"])
            goal = tuple(int(x) for x in payload["goal"])
            rules = tuple(
                Rule(
                    id=str(r["id"]),
                    consume=tuple(int(x) for x in r["consume"]),
                    produce=tuple(int(x) for x in r["produce"]),
                )
                for r in payload["rules"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorldError(f"字段缺失或类型错误: {exc}") from exc
        world = World(
            symbols=symbols,
            initial=initial,
            goal=goal,
            rules=rules,
            world_id=str(payload.get("world_id", "")),
            group_id=str(payload.get("group_id", "")),
            meta=dict(payload.get("meta") or {}),
        )
        validate_world(world)
        return world


def validate_world(world: World) -> None:
    """检查第 2.1 节的首版约束。不满足即报错。"""
    n = len(world.symbols)
    if n == 0:
        raise WorldError("资源列表为空")
    if len(set(world.symbols)) != n:
        raise WorldError("资源符号重复")
    for sym in world.symbols:
        if not sym or any(ch.isspace() for ch in sym):
            raise WorldError(f"资源符号非法: {sym!r}")
    for name, vec in (("initial", world.initial), ("goal", world.goal)):
        if len(vec) != n:
            raise WorldError(f"{name} 长度 {len(vec)} 与资源数 {n} 不符")
        if any(v < 0 for v in vec):
            raise WorldError(f"{name} 含负计数")
    if sum(world.initial) < 1:
        raise WorldError("初态总资源量为 0")
    if sum(world.goal) < 1:
        raise WorldError("目标总资源量为 0")

    seen: set[str] = set()
    last_index = -1
    for rule in world.rules:
        if not RULE_ID_RE.match(rule.id):
            raise WorldError(f"规则 ID 非法: {rule.id!r}")
        if rule.id in seen:
            raise WorldError(f"规则 ID 重复: {rule.id}")
        seen.add(rule.id)
        if rule.index <= last_index:
            raise WorldError(f"规则未按数字 ID 升序排列: {rule.id}")
        last_index = rule.index
        if len(rule.consume) != n or len(rule.produce) != n:
            raise WorldError(f"{rule.id} 的向量长度与资源数不符")
        if any(v < 0 for v in rule.consume) or any(v < 0 for v in rule.produce):
            raise WorldError(f"{rule.id} 含负计数")
        if sum(rule.consume) < 1:
            raise WorldError(f"{rule.id} 输入为空")
        if sum(rule.produce) < 1:
            raise WorldError(f"{rule.id} 输出为空")
        if sum(rule.produce) > sum(rule.consume):
            raise WorldError(f"{rule.id} 输出总量大于输入总量")
        if rule.consume == rule.produce:
            raise WorldError(f"{rule.id} 为原地不变规则")


# --------------------------------------------------------------------------- #
# 向量的两种文本编码（附录 B）
# --------------------------------------------------------------------------- #

VECTOR_DENSE = "dense"
VECTOR_SPARSE = "sparse"
VECTOR_FORMATS = (VECTOR_DENSE, VECTOR_SPARSE)
EMPTY_VECTOR_TOKEN = "-"

SPARSE_TERM_RE = re.compile(r"^([^\s:]+):(0|[1-9][0-9]*)$")


def format_vector(
    vec: Sequence[int], symbols: Sequence[str] | None = None, fmt: str = VECTOR_DENSE
) -> str:
    """把计数向量渲染为事件行中的文本。

    `dense` 逐项列出全部计数；`sparse` 只列非零项，按资源顺序升序，全零写 `-`。
    两种编码都是规范的：给定向量只有一种渲染结果。
    """
    if fmt == VECTOR_DENSE:
        return " ".join(str(v) for v in vec)
    if fmt != VECTOR_SPARSE:
        raise ValueError(f"未知向量编码: {fmt}")
    if symbols is None:
        raise ValueError("sparse 编码需要资源符号表")
    terms = [f"{symbols[i]}:{v}" for i, v in enumerate(vec) if v]
    return " ".join(terms) if terms else EMPTY_VECTOR_TOKEN


def detect_vector_format(tokens: Sequence[str]) -> str:
    """按首个记号判定编码。数字开头为 dense，含冒号或等于 `-` 为 sparse。"""
    if not tokens:
        return VECTOR_DENSE
    return VECTOR_SPARSE if (":" in tokens[0] or tokens[0] == EMPTY_VECTOR_TOKEN) else VECTOR_DENSE


def parse_vector(
    tokens: Sequence[str], symbols: Sequence[str]
) -> tuple[Vector | None, str, str | None]:
    """解析向量记号序列，返回 (向量, 编码, 错误码)。

    dense 要求项数等于资源数；sparse 要求符号已知、不重复、按资源顺序升序、计数为正。
    """
    n = len(symbols)
    fmt = detect_vector_format(tokens)
    if fmt == VECTOR_DENSE:
        if len(tokens) != n:
            return None, fmt, VECTOR_LENGTH
        out = []
        for tok in tokens:
            if not INT_RE.match(tok):
                return None, fmt, PARSE_ERROR
            out.append(int(tok))
        return tuple(out), fmt, None

    index = {name: i for i, name in enumerate(symbols)}
    out = [0] * n
    if list(tokens) == [EMPTY_VECTOR_TOKEN]:
        return tuple(out), fmt, None
    last = -1
    for tok in tokens:
        match = SPARSE_TERM_RE.match(tok)
        if not match:
            return None, fmt, PARSE_ERROR
        name, count = match.group(1), int(match.group(2))
        if name not in index:
            return None, fmt, VECTOR_SYMBOL_UNKNOWN
        if count == 0:
            # 零项必须省略，否则同一向量会有多种写法。
            return None, fmt, PARSE_ERROR
        pos = index[name]
        if out[pos]:
            return None, fmt, VECTOR_SYMBOL_DUPLICATE
        if pos <= last:
            return None, fmt, VECTOR_SYMBOL_ORDER
        last = pos
        out[pos] = count
    return tuple(out), fmt, None


# --------------------------------------------------------------------------- #
# 状态转移与需求回归
# --------------------------------------------------------------------------- #


def dominates(left: Sequence[int], right: Sequence[int]) -> bool:
    """逐坐标判断 left >= right。"""
    return all(a >= b for a, b in zip(left, right))


def applicable(state: Vector, rule: Rule) -> bool:
    return dominates(state, rule.consume)


def step(state: Vector, rule: Rule) -> Vector | None:
    """正向转移 T(s,r)=s-u_r+v_r；不可执行返回 None。"""
    if not applicable(state, rule):
        return None
    return tuple(s - c + p for s, c, p in zip(state, rule.consume, rule.produce))


def regress(demand: Vector, rule: Rule, total_units: int) -> Vector | None:
    """反向回归 Regress(d,r)=u_r+max(d-v_r,0)；总需求超过 m 返回 None。"""
    out = tuple(c + max(d - p, 0) for d, c, p in zip(demand, rule.consume, rule.produce))
    if sum(out) > total_units:
        return None
    return out


def meets_goal(state: Vector, goal: Vector) -> bool:
    return dominates(state, goal)


def demand_met(demand: Vector, initial: Vector) -> bool:
    return dominates(initial, demand)


def world_step(world: World, state: Vector, rule_id: str) -> Vector | None:
    rule = world.rule(rule_id)
    if rule is None:
        raise WorldError(f"未知规则: {rule_id}")
    return step(state, rule)


def world_regress(world: World, demand: Vector, rule_id: str) -> Vector | None:
    rule = world.rule(rule_id)
    if rule is None:
        raise WorldError(f"未知规则: {rule_id}")
    return regress(demand, rule, world.total_units)


# --------------------------------------------------------------------------- #
# 最终方案验证
# --------------------------------------------------------------------------- #


@dataclass
class PlanCheck:
    """方案验证结果与资源开销记账。

    `valid` 只表示可执行且达成目标。冗余步骤按第 2.2 节是合法的，其代价由
    `first_goal_step`、`state_revisits` 与资源记账反映，不影响 `valid`。
    """

    valid: bool
    steps: int
    final_state: Vector | None
    error: Issue | None = None
    first_goal_step: int | None = None
    units_initial: int = 0
    units_final: int = 0
    state_revisits: int = 0
    units_monotone: bool = True

    @property
    def units_consumed(self) -> int:
        return self.units_initial - self.units_final

    @property
    def steps_after_goal(self) -> int | None:
        if self.first_goal_step is None:
            return None
        return self.steps - self.first_goal_step

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "steps": self.steps,
            "final_state": list(self.final_state) if self.final_state is not None else None,
            "error": self.error.to_dict() if self.error else None,
            "first_goal_step": self.first_goal_step,
            "steps_after_goal": self.steps_after_goal,
            "units_initial": self.units_initial,
            "units_final": self.units_final,
            "units_consumed": self.units_consumed,
            "state_revisits": self.state_revisits,
            "units_monotone": self.units_monotone,
        }


def verify_plan(
    world: World, plan: Sequence[str], max_steps: int | None = None
) -> PlanCheck:
    """从 s_0 逐步执行规则序列，全部成功后检查目标，并记录资源开销。

    不要求最后一步首次达到目标，不禁止冗余步骤，不检查最优性。`line` 为 1 起的步序号。
    `max_steps` 为方案长度上限，超出时直接返回 `PLAN_TOO_LONG`，不执行任何步骤。
    """
    units_initial = sum(world.initial)
    base = PlanCheck(
        valid=False,
        steps=0,
        final_state=world.initial,
        units_initial=units_initial,
        units_final=units_initial,
    )
    if max_steps is not None and len(plan) > max_steps:
        base.error = Issue(PLAN_TOO_LONG, None, f"方案 {len(plan)} 步，上限 {max_steps}")
        return base

    state = world.initial
    seen = {state}
    revisits = 0
    first_goal: int | None = 0 if meets_goal(state, world.goal) else None
    monotone = True

    for i, rule_id in enumerate(plan, start=1):
        rule = world.rule(rule_id)
        if rule is None:
            base.steps = i - 1
            base.final_state = state
            base.units_final = sum(state)
            base.first_goal_step = first_goal
            base.state_revisits = revisits
            base.units_monotone = monotone
            base.error = Issue(UNKNOWN_RULE, i, rule_id)
            return base
        nxt = step(state, rule)
        if nxt is None:
            missing = [
                world.symbols[k]
                for k in range(world.n_resources)
                if state[k] < rule.consume[k]
            ]
            base.steps = i - 1
            base.final_state = state
            base.units_final = sum(state)
            base.first_goal_step = first_goal
            base.state_revisits = revisits
            base.units_monotone = monotone
            base.error = Issue(
                INSUFFICIENT_RESOURCE, i, f"{rule_id} 缺少 {','.join(missing)}"
            )
            return base
        if sum(nxt) > sum(state):
            # 首版规则约束保证总量不增加；若触发说明世界定义违反契约。
            monotone = False
        state = nxt
        if state in seen:
            revisits += 1
        else:
            seen.add(state)
        if first_goal is None and meets_goal(state, world.goal):
            first_goal = i

    base.steps = len(plan)
    base.final_state = state
    base.units_final = sum(state)
    base.first_goal_step = first_goal
    base.state_revisits = revisits
    base.units_monotone = monotone
    if not meets_goal(state, world.goal):
        base.error = Issue(GOAL_NOT_MET, len(plan), None)
        return base
    base.valid = True
    return base


# --------------------------------------------------------------------------- #
# 可达性、修复代价与步骤分类（第 6.9 节）
# --------------------------------------------------------------------------- #

DEFAULT_CRITICALITY_STATES = 20_000

REACHABLE = "reachable"
UNREACHABLE = "unreachable"
UNDETERMINED = "undetermined"

KIND_DOOMED = "doomed"
KIND_FORCED = "forced"
KIND_FREE = "free"
KIND_WASTEFUL = "wasteful"
KIND_CRITICAL = "critical"
KIND_UNKNOWN = "unknown"


def shortest_distance(
    world: World, state: Vector, max_states: int = DEFAULT_CRITICALITY_STATES
) -> tuple[str, int | None]:
    """从 state 到目标的最少步数。

    返回 (状态, 步数)。状态为 `reachable` 时步数有效；`unreachable` 表示穷举完毕仍未
    达到目标；`undetermined` 表示访问状态数超过 max_states，步数为 None。
    按层展开，因此首次达到目标的深度即最少步数。
    """
    from collections import deque

    if meets_goal(state, world.goal):
        return REACHABLE, 0
    depth = {state: 0}
    queue = deque([state])
    while queue:
        current = queue.popleft()
        d = depth[current]
        for rule in world.rules:
            nxt = step(current, rule)
            if nxt is None or nxt in depth:
                continue
            if meets_goal(nxt, world.goal):
                return REACHABLE, d + 1
            if len(depth) >= max_states:
                return UNDETERMINED, None
            depth[nxt] = d + 1
            queue.append(nxt)
    return UNREACHABLE, None


def goal_reachable(world: World, state: Vector, max_states: int = DEFAULT_CRITICALITY_STATES):
    status, _ = shortest_distance(world, state, max_states)
    if status == UNDETERMINED:
        return None
    return status == REACHABLE


@dataclass
class Deviation:
    """在某一步改用另一条可执行规则的后果。"""

    rule_id: str
    status: str
    waste: int | None
    suffix_valid: bool | None

    @property
    def fatal(self) -> bool:
        return self.status == UNREACHABLE

    @property
    def undetermined(self) -> bool:
        return self.status == UNDETERMINED

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "status": self.status,
            "waste": self.waste,
            "suffix_valid": self.suffix_valid,
        }


@dataclass
class StepOutcome:
    """单步的可执行规则数与全部替代规则的后果分布。"""

    applicable: int
    best_distance: int | None
    deviations: list[Deviation]
    best_status: str = REACHABLE
    taken_status: str = REACHABLE
    taken_waste: int | None = None

    @property
    def alternatives(self) -> int:
        return len(self.deviations)

    @property
    def fatal(self) -> int:
        return sum(1 for d in self.deviations if d.fatal)

    @property
    def wasteful(self) -> int:
        return sum(1 for d in self.deviations if d.waste is not None and d.waste > 0)

    @property
    def free(self) -> int:
        return sum(1 for d in self.deviations if d.waste == 0)

    @property
    def undetermined(self) -> int:
        return sum(1 for d in self.deviations if d.undetermined)

    @property
    def suffix_valid(self) -> int:
        return sum(1 for d in self.deviations if d.suffix_valid)

    @property
    def max_waste(self) -> int | None:
        values = [d.waste for d in self.deviations if d.waste is not None]
        return max(values) if values else None

    @property
    def kind(self) -> str:
        """强制步、自由步、浪费步、关键步；另有两个只在分析无效方案时出现的类别。"""
        if (
            self.undetermined
            or self.best_status == UNDETERMINED
            or self.taken_status == UNDETERMINED
        ):
            return KIND_UNKNOWN
        if self.best_status == UNREACHABLE:
            # 该步起点已经不可达，损失发生在更早的步骤。
            return KIND_DOOMED
        if self.applicable <= 1:
            return KIND_FORCED
        # 类别只取决于该状态上是否存在必错选择，与实际选了哪一条无关。
        if self.fatal or self.taken_status == UNREACHABLE:
            return KIND_CRITICAL
        if self.wasteful:
            return KIND_WASTEFUL
        return KIND_FREE

    def to_dict(self) -> dict:
        return {
            "applicable": self.applicable,
            "alternatives": self.alternatives,
            "best_distance": self.best_distance,
            "best_status": self.best_status,
            "taken_status": self.taken_status,
            "taken_waste": self.taken_waste,
            "kind": self.kind,
            "fatal": self.fatal,
            "wasteful": self.wasteful,
            "free": self.free,
            "undetermined": self.undetermined,
            "suffix_valid": self.suffix_valid,
            "max_waste": self.max_waste,
        }


@dataclass
class PlanOutcome:
    steps: list[StepOutcome]
    states_limit: int

    def count(self, kind: str) -> int:
        return sum(1 for s in self.steps if s.kind == kind)

    @property
    def critical_steps(self) -> int:
        return self.count(KIND_CRITICAL)

    @property
    def wasteful_steps(self) -> int:
        return self.count(KIND_WASTEFUL)

    @property
    def free_steps(self) -> int:
        return self.count(KIND_FREE)

    @property
    def forced_steps(self) -> int:
        return self.count(KIND_FORCED)

    @property
    def unknown_steps(self) -> int:
        return self.count(KIND_UNKNOWN)

    @property
    def doomed_steps(self) -> int:
        return self.count(KIND_DOOMED)

    @property
    def critical_positions(self) -> list[float]:
        n = len(self.steps)
        if not n:
            return []
        return [
            round((i + 1) / n, 4)
            for i, s in enumerate(self.steps)
            if s.kind == KIND_CRITICAL
        ]

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "forced_steps": self.forced_steps,
            "free_steps": self.free_steps,
            "wasteful_steps": self.wasteful_steps,
            "critical_steps": self.critical_steps,
            "doomed_steps": self.doomed_steps,
            "unknown_steps": self.unknown_steps,
            "critical_positions": self.critical_positions,
            "states_limit": self.states_limit,
        }

    @staticmethod
    def from_dict(payload: dict) -> "PlanOutcome":
        steps = []
        for item in payload.get("steps", []):
            steps.append(
                StepOutcome(
                    applicable=int(item["applicable"]),
                    best_distance=item.get("best_distance"),
                    deviations=[],
                    best_status=str(item.get("best_status", REACHABLE)),
                )
            )
        return PlanOutcome(
            steps=steps, states_limit=int(payload.get("states_limit", DEFAULT_CRITICALITY_STATES))
        )


def step_outcome(
    world: World,
    state: Vector,
    taken: str,
    suffix: Sequence[str],
    max_states: int = DEFAULT_CRITICALITY_STATES,
) -> StepOutcome:
    """枚举该状态上的全部可执行规则，给出每条替代规则的后果与修复代价。

    `waste` 为改用该规则后到目标的最少步数，减去从当前状态出发的最少步数加一，即相对
    最优走法多出的步数。目标不可达时 `waste` 为 None 且 `status` 为 `unreachable`。
    """
    applicable_rules = [r for r in world.rules if applicable(state, r)]
    base_status, base_distance = shortest_distance(world, state, max_states)

    def consequence(rule: Rule) -> tuple[str, int | None, bool | None]:
        nxt = step(state, rule)
        assert nxt is not None
        status, distance = shortest_distance(world, nxt, max_states)
        if status != REACHABLE or base_status != REACHABLE:
            return status, None, None
        assert base_distance is not None and distance is not None
        probe = World(
            symbols=world.symbols,
            initial=nxt,
            goal=world.goal,
            rules=world.rules,
            world_id="probe",
        )
        return status, distance - (base_distance - 1), verify_plan(probe, suffix).valid

    deviations: list[Deviation] = []
    taken_status = REACHABLE
    taken_waste: int | None = None
    for rule in applicable_rules:
        status, waste, suffix_valid = consequence(rule)
        if rule.id == taken:
            taken_status, taken_waste = status, waste
            continue
        deviations.append(
            Deviation(rule_id=rule.id, status=status, waste=waste, suffix_valid=suffix_valid)
        )
    if taken not in {r.id for r in applicable_rules}:
        # 所取规则在该状态不可执行，其后果由调用方的方案验证给出。
        taken_status = UNREACHABLE if base_status == UNREACHABLE else REACHABLE
    return StepOutcome(
        applicable=len(applicable_rules),
        best_distance=base_distance,
        deviations=deviations,
        best_status=base_status,
        taken_status=taken_status,
        taken_waste=taken_waste,
    )


def plan_outcome(
    world: World, plan: Sequence[str], max_states: int = DEFAULT_CRITICALITY_STATES
) -> PlanOutcome:
    """沿给定方案逐步计算步骤分类与修复代价。方案必须可执行。"""
    steps: list[StepOutcome] = []
    state = world.initial
    for t, rule_id in enumerate(plan):
        steps.append(step_outcome(world, state, rule_id, list(plan[t + 1 :]), max_states))
        rule = world.rule(rule_id)
        if rule is None:
            raise WorldError(f"方案含未知规则: {rule_id}")
        nxt = step(state, rule)
        if nxt is None:
            raise WorldError(f"方案在 {rule_id} 处不可执行")
        state = nxt
    return PlanOutcome(steps=steps, states_limit=max_states)


# --------------------------------------------------------------------------- #
# 标识哈希（附录 D）
# --------------------------------------------------------------------------- #


def _sha256_hex(payload: str, nbytes: int) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[: nbytes * 2]


def canonical_world_payload(
    symbols: Sequence[str], initial: Sequence[int], goal: Sequence[int], rules: Sequence[Rule]
) -> str:
    """world_id 的规范序列化：键序固定、无空格、按最终顺序。"""
    return json.dumps(
        {
            "symbols": list(symbols),
            "initial": list(initial),
            "goal": list(goal),
            "rules": [[r.id, list(r.consume), list(r.produce)] for r in rules],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def make_world_id(
    symbols: Sequence[str], initial: Sequence[int], goal: Sequence[int], rules: Sequence[Rule]
) -> str:
    return "w_" + _sha256_hex(canonical_world_payload(symbols, initial, goal, rules), 12)


def make_group_id(construction_payload: str) -> str:
    """group_id 由打乱符号与规则顺序之前的底层构造决定。"""
    return "g_" + _sha256_hex(construction_payload, 12)


def structure_fingerprint(world: World, rounds: int = 3) -> str:
    """在符号置换与规则置换下不变的结构指纹（附录 D 第 1–3 步）。

    资源节点初始颜色为 (s0[a], g[a])，规则节点初始颜色为 (|u|_1, |v|_1)；
    资源 a 与规则 r 之间的边标签为 (u_r[a], v_r[a])，两者为零时无边。
    """
    n = world.n_resources
    res_color = [_hash_color(("A", world.initial[a], world.goal[a])) for a in range(n)]
    rule_color = [_hash_color(("R", sum(r.consume), sum(r.produce))) for r in world.rules]

    # 邻接：资源 -> [(规则下标, 边标签)]，规则 -> [(资源下标, 边标签)]
    res_adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in range(n)]
    rule_adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in world.rules]
    for j, rule in enumerate(world.rules):
        for a in range(n):
            label = (rule.consume[a], rule.produce[a])
            if label == (0, 0):
                continue
            res_adj[a].append((j, label))
            rule_adj[j].append((a, label))

    for _ in range(rounds):
        new_res = [
            _hash_color((res_color[a], tuple(sorted((lab, rule_color[j]) for j, lab in res_adj[a]))))
            for a in range(n)
        ]
        new_rule = [
            _hash_color((rule_color[j], tuple(sorted((lab, res_color[a]) for a, lab in rule_adj[j]))))
            for j in range(len(world.rules))
        ]
        res_color, rule_color = new_res, new_rule

    digest = json.dumps(
        {"res": sorted(res_color), "rule": sorted(rule_color)}, separators=(",", ":")
    )
    return _sha256_hex(digest, 16)


def _hash_color(value) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:16]


def refined_colors(world: World, rounds: int = 3) -> tuple[list[str], list[str]]:
    """返回细化后的资源颜色与规则颜色，供同构检查分组。"""
    n = world.n_resources
    res_color = [_hash_color(("A", world.initial[a], world.goal[a])) for a in range(n)]
    rule_color = [_hash_color(("R", sum(r.consume), sum(r.produce))) for r in world.rules]
    res_adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in range(n)]
    rule_adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in world.rules]
    for j, rule in enumerate(world.rules):
        for a in range(n):
            label = (rule.consume[a], rule.produce[a])
            if label == (0, 0):
                continue
            res_adj[a].append((j, label))
            rule_adj[j].append((a, label))
    for _ in range(rounds):
        new_res = [
            _hash_color((res_color[a], tuple(sorted((lab, rule_color[j]) for j, lab in res_adj[a]))))
            for a in range(n)
        ]
        new_rule = [
            _hash_color((rule_color[j], tuple(sorted((lab, res_color[a]) for a, lab in rule_adj[j]))))
            for j in range(len(world.rules))
        ]
        res_color, rule_color = new_res, new_rule
    return res_color, rule_color


ISO_STEP_LIMIT = 200_000


def isomorphic(
    left: World, right: World, step_limit: int = ISO_STEP_LIMIT
) -> tuple[bool | None, int]:
    """精确结构同构检查（附录 D 第 5 步）。

    返回 (结果, 回溯步数)。结果为 None 表示超出回溯步数上限，调用方记
    `iso_check_unresolved` 并按结构独立处理。
    """
    if left.n_resources != right.n_resources or len(left.rules) != len(right.rules):
        return False, 0
    lr_color, lu_color = refined_colors(left)
    rr_color, ru_color = refined_colors(right)
    if sorted(lr_color) != sorted(rr_color) or sorted(lu_color) != sorted(ru_color):
        return False, 0

    n = left.n_resources
    # 资源候选按颜色分组；规则映射由资源映射推出后校验。
    candidates = [[b for b in range(n) if rr_color[b] == lr_color[a]] for a in range(n)]
    order = sorted(range(n), key=lambda a: len(candidates[a]))
    right_rule_key = {}
    for j, rule in enumerate(right.rules):
        right_rule_key.setdefault((rule.consume, rule.produce), []).append(j)

    mapping = [-1] * n
    used = [False] * n
    steps = 0

    def consistent_prefix(pos: int) -> bool:
        """已映射的资源下标上，检查初态/目标计数一致。"""
        a = order[pos]
        b = mapping[a]
        return left.initial[a] == right.initial[b] and left.goal[a] == right.goal[b]

    def rules_match() -> bool:
        perm = mapping
        pool = {k: list(v) for k, v in right_rule_key.items()}
        for rule in left.rules:
            consume = [0] * n
            produce = [0] * n
            for a in range(n):
                consume[perm[a]] = rule.consume[a]
                produce[perm[a]] = rule.produce[a]
            key = (tuple(consume), tuple(produce))
            bucket = pool.get(key)
            if not bucket:
                return False
            bucket.pop()
        return True

    def backtrack(pos: int) -> bool | None:
        nonlocal steps
        if pos == n:
            return rules_match()
        for b in candidates[order[pos]]:
            if used[b]:
                continue
            steps += 1
            if steps > step_limit:
                return None
            mapping[order[pos]] = b
            used[b] = True
            if consistent_prefix(pos):
                result = backtrack(pos + 1)
                if result is None:
                    return None
                if result:
                    return True
            used[b] = False
            mapping[order[pos]] = -1
        return False

    result = backtrack(0)
    return result, steps


# --------------------------------------------------------------------------- #
# JSONL 辅助
# --------------------------------------------------------------------------- #


def iter_jsonl(path, *, allow_missing: bool = False) -> Iterable[dict]:
    import os

    if allow_missing and not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorldError(f"{path}:{lineno} JSON 解析失败: {exc}") from exc


def dump_jsonl_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def load_worlds(path) -> dict[str, World]:
    worlds: dict[str, World] = {}
    for payload in iter_jsonl(path):
        world = World.from_dict(payload)
        if not world.world_id:
            raise WorldError(f"{path} 中存在缺少 world_id 的记录")
        if world.world_id in worlds:
            raise WorldError(f"world_id 重复: {world.world_id}")
        worlds[world.world_id] = world
    if not worlds:
        raise WorldError(f"{path} 没有可用记录")
    return worlds

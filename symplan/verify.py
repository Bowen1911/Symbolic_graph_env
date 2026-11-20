"""方案、完整搜索过程与前缀的验证。

验证器独立重放事件，不信任 Teacher 输出的状态，也不比较字符串是否与 Teacher 原样
相同。plan 验证路径与 search 验证路径分开（第 12.1 节）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import env, search
from .env import Issue, Vector, World

KIND_PLAN = "plan"
KIND_SEARCH = "search"
KIND_PREFIX = "prefix"
KINDS = (KIND_PLAN, KIND_SEARCH, KIND_PREFIX)
VERIFICATION_POLICY = "independent_valid_plan_v2"

OUTCOME_SOLVED = "solved"
OUTCOME_BUDGET_EXHAUSTED = "budget_exhausted"
OUTCOME_STOPPED_EARLY = "stopped_early"
OUTCOME_FRONTIER_EMPTY = "frontier_empty"
OUTCOME_INVALID = "invalid"

PARSE_COMPLETE = "complete"
PARSE_PREFIX = "prefix"
PARSE_PENDING = "pending_event"
PARSE_ERROR_STATUS = "error"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
NONTHINK_PREFIX = "<think>\n\n</think>\n\n"
THINK_PREFIX = "<think>\n"

EVENT_KEYWORDS = ("EXPAND", "NEW", "DUP", "CLOSE", "GOAL", "PLAN", "END")

# 验证成本上限。模型输出不可信，必须在解析与重放之前设界。
DEFAULT_MAX_PLAN_STEPS = 512
DEFAULT_MAX_EVENTS = 200_000


# --------------------------------------------------------------------------- #
# 文本切分
# --------------------------------------------------------------------------- #


def restore_prefill(text: str, prefill: str | None) -> str:
    """推理只返回后缀时恢复已知前缀；已含 think 块时不重复添加。"""
    if prefill is None:
        return text
    if text.lstrip().startswith(THINK_OPEN):
        return text
    if prefill == "nonthink":
        return NONTHINK_PREFIX + text
    if prefill == "think":
        return THINK_PREFIX + text
    raise ValueError(f"未知 --prefill 取值: {prefill}")


@dataclass
class SplitText:
    lines: list[str]
    inner_count: int
    had_think: bool
    think_closed: bool
    plan_inside_think: bool


def split_assistant_text(text: str) -> SplitText:
    """把 assistant 文本切成事件行序列。

    think 块的边界不改变事件序列，只用于格式检查：PLAN/END 必须出现在块外。
    """
    had_think = False
    think_closed = False
    inner_text = ""
    outer_text = text
    stripped = text.lstrip()
    if stripped.startswith(THINK_OPEN):
        had_think = True
        body = stripped[len(THINK_OPEN) :]
        idx = body.find(THINK_CLOSE)
        if idx < 0:
            inner_text, outer_text = body, ""
        else:
            think_closed = True
            inner_text = body[:idx]
            outer_text = body[idx + len(THINK_CLOSE) :]

    inner = [ln for ln in inner_text.split("\n") if ln.strip()]
    outer = [ln for ln in outer_text.split("\n") if ln.strip()]
    plan_inside = any(ln.startswith("PLAN") or ln == "END" for ln in inner)
    return SplitText(
        lines=inner + outer,
        inner_count=len(inner),
        had_think=had_think,
        think_closed=think_closed,
        plan_inside_think=plan_inside,
    )


# --------------------------------------------------------------------------- #
# 事件解析（附录 B）
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    kind: str
    line: int
    raw: str
    node: str | None = None
    rule: str | None = None
    keyword: str | None = None
    vector: Vector | None = None
    vector_format: str | None = None
    plan: list[str] | None = None

    @property
    def node_index(self) -> int:
        assert self.node is not None
        return int(self.node[1:])


_SCHEMAS: dict[str, tuple[str, ...]] = {
    # 位置类型：LIT:<text> 字面量、NODE、RULE、KW、COUNTS、RULES
    "EXPAND": ("NODE", "KW", "COUNTS"),
    "NEW": ("NODE", "LIT:VIA", "RULE", "KW", "COUNTS"),
    "DUP": ("NODE", "LIT:VIA", "RULE"),
    "CLOSE": ("NODE",),
    "GOAL": ("NODE",),
    "PLAN": ("RULES",),
    "END": (),
}


def parse_event(
    line: str, lineno: int, n_resources: int, symbols: Sequence[str] | None = None
) -> Event | Issue:
    """解析单行事件。不满足附录 B 的文法返回 Issue。

    向量的两种编码按首个记号自动判别；`symbols` 缺省时只接受 dense 编码。
    """
    if symbols is None:
        symbols = tuple(f"s{i}" for i in range(n_resources))
    if line != line.strip() or "  " in line:
        return Issue(env.PARSE_ERROR, lineno, "行首尾空白或连续空格")
    tokens = line.split(" ")
    head = tokens[0]
    if head not in _SCHEMAS:
        return Issue(env.PARSE_ERROR, lineno, f"未知事件 {head!r}")
    schema = _SCHEMAS[head]
    rest = tokens[1:]
    event = Event(kind=head, line=lineno, raw=line)
    pos = 0
    for slot in schema:
        if slot == "COUNTS":
            counts = rest[pos:]
            if not counts:
                return Issue(env.PARSE_ERROR, lineno, f"{head} 缺少向量")
            vector, fmt, code = env.parse_vector(counts, symbols)
            if code is not None:
                detail = (
                    f"向量项数 {len(counts)}，资源数 {n_resources}"
                    if code == env.VECTOR_LENGTH
                    else " ".join(counts)
                )
                return Issue(code, lineno, detail)
            event.vector = vector
            event.vector_format = fmt
            pos = len(rest)
            continue
        if slot == "RULES":
            rules = rest[pos:]
            for tok in rules:
                if not env.RULE_ID_RE.match(tok):
                    return Issue(env.PARSE_ERROR, lineno, f"非法规则 ID {tok!r}")
            event.plan = list(rules)
            pos = len(rest)
            continue
        if pos >= len(rest):
            return Issue(env.PARSE_ERROR, lineno, f"{head} 缺少字段")
        tok = rest[pos]
        if slot.startswith("LIT:"):
            if tok != slot[4:]:
                return Issue(env.PARSE_ERROR, lineno, f"期望 {slot[4:]}，实际 {tok!r}")
        elif slot == "NODE":
            if not env.NODE_ID_RE.match(tok):
                return Issue(env.PARSE_ERROR, lineno, f"非法节点 ID {tok!r}")
            event.node = tok
        elif slot == "RULE":
            if not env.RULE_ID_RE.match(tok):
                return Issue(env.PARSE_ERROR, lineno, f"非法规则 ID {tok!r}")
            event.rule = tok
        elif slot == "KW":
            if tok not in ("STATE", "NEED"):
                return Issue(env.PARSE_ERROR, lineno, f"期望 STATE 或 NEED，实际 {tok!r}")
            event.keyword = tok
        pos += 1
    if pos != len(rest):
        return Issue(env.PARSE_ERROR, lineno, f"{head} 多余字段")
    return event


def is_partial_event(line: str, n_resources: int) -> bool:
    """判断一行是否为某个合法事件的真前缀（附录 B 约束 6）。"""
    if line.endswith(" ") or "  " in line or line != line.lstrip():
        return False
    tokens = line.split(" ")
    head = tokens[0]
    if len(tokens) == 1:
        return any(kw.startswith(head) and kw != head for kw in EVENT_KEYWORDS)
    if head not in _SCHEMAS:
        return False
    schema = _SCHEMAS[head]
    rest = tokens[1:]
    pos = 0
    for i, tok in enumerate(rest):
        last = i == len(rest) - 1
        if pos >= len(schema):
            return False
        slot = schema[pos]
        if slot == "COUNTS":
            remaining = rest[i:]
            if len(remaining) > n_resources:
                return False
            for j, c in enumerate(remaining):
                is_tail = j == len(remaining) - 1
                if not env.INT_RE.match(c) and not (is_tail and c.isdigit()):
                    return False
            return len(remaining) < n_resources or not _complete_counts(remaining, n_resources)
        if slot == "RULES":
            for c in rest[i:]:
                if not env.RULE_ID_RE.match(c) and not _rule_prefix(c):
                    return False
            return True
        ok = _slot_match(slot, tok, partial=last)
        if not ok:
            return False
        pos += 1
    return pos < len(schema)


def _complete_counts(tokens: Sequence[str], n_resources: int) -> bool:
    return len(tokens) == n_resources and all(env.INT_RE.match(t) for t in tokens)


def _rule_prefix(tok: str) -> bool:
    return tok == "r" or (tok.startswith("r") and tok[1:].isdigit())


def _slot_match(slot: str, tok: str, *, partial: bool) -> bool:
    if slot.startswith("LIT:"):
        literal = slot[4:]
        return tok == literal or (partial and literal.startswith(tok))
    if slot == "NODE":
        if env.NODE_ID_RE.match(tok):
            return True
        return partial and (tok == "n" or (tok.startswith("n") and tok[1:].isdigit()))
    if slot == "RULE":
        if env.RULE_ID_RE.match(tok):
            return True
        return partial and _rule_prefix(tok)
    if slot == "KW":
        if tok in ("STATE", "NEED"):
            return True
        return partial and ("STATE".startswith(tok) or "NEED".startswith(tok))
    return False


# --------------------------------------------------------------------------- #
# 事件重放
# --------------------------------------------------------------------------- #


@dataclass
class ReplayState:
    checked_nodes: int = 0
    goal_node: int | None = None
    plan: list[str] | None = None
    plan_consistent: bool | None = None
    finished: bool = False
    error: Issue | None = None
    consumed: int = 0
    frontier_at_error: int | None = None
    frontier_remaining: int = 0


class Replay:
    """按给定方向与调度重放事件。

    schedule 为 `any` 时只要求 EXPAND 的节点在 frontier 中，用于通用过程检查；
    `fifo`/`lifo` 用于固定模式判别。
    """

    def __init__(self, world: World, direction: str, schedule: str):
        self.world = world
        self.direction = direction
        self.forward = direction == "forward"
        self.schedule = schedule
        self.keyword = search.vector_keyword(direction)
        root: Vector = world.initial if self.forward else world.goal
        self.vectors: list[Vector] = [root]
        self.parent: list[int] = [-1]
        self.via: list[str | None] = [None]
        self.visited: dict[Vector, int] = {root: 0}
        self.frontier: list[int] = [0]
        self.pending_new: list[int] = []
        self.current: int | None = None
        self.expected: list[tuple[str, int, str, Vector | None]] = []
        self.cursor = 0
        self.phase = "need_expand"
        self.state = ReplayState()

    # -- 对外 -------------------------------------------------------------- #

    def feed(self, event: Event) -> Issue | None:
        if self.state.error is not None:
            return self.state.error
        issue = self._dispatch(event)
        if issue is not None:
            self.state.error = issue
            if event.kind == "EXPAND":
                self.state.frontier_at_error = len(self.frontier)
        else:
            self.state.consumed += 1
        return issue

    def run(self, events: Sequence[Event]) -> ReplayState:
        for event in events:
            if self.feed(event) is not None:
                break
        self.state.finished = self.phase == "done"
        self.state.frontier_remaining = len(self.frontier) + len(self.pending_new)
        return self.state

    # -- 事件分派 ----------------------------------------------------------- #

    def _dispatch(self, event: Event) -> Issue | None:
        if event.keyword is not None and event.keyword != self.keyword:
            return Issue(
                env.DIRECTION_MISMATCH,
                event.line,
                f"期望 {self.keyword}，实际 {event.keyword}",
            )
        handler = {
            "EXPAND": self._on_expand,
            "NEW": self._on_new,
            "DUP": self._on_dup,
            "CLOSE": self._on_close,
            "GOAL": self._on_goal,
            "PLAN": self._on_plan,
            "END": self._on_end,
        }[event.kind]
        return handler(event)

    def _on_expand(self, event: Event) -> Issue | None:
        if self.phase != "need_expand":
            return Issue(env.UNEXPECTED_EVENT, event.line, f"EXPAND 出现在 {self.phase}")
        idx = event.node_index
        if idx >= len(self.vectors):
            return Issue(env.UNKNOWN_NODE, event.line, event.node)
        if idx not in self.frontier:
            return Issue(env.BAD_FRONTIER_CHOICE, event.line, f"{event.node} 不在 frontier")
        if self.schedule == "fifo" and idx != self.frontier[0]:
            return Issue(
                env.BAD_FRONTIER_CHOICE, event.line, f"FIFO 应取 n{self.frontier[0]}"
            )
        if self.schedule == "lifo" and idx != self.frontier[-1]:
            return Issue(
                env.BAD_FRONTIER_CHOICE, event.line, f"LIFO 应取 n{self.frontier[-1]}"
            )
        if event.vector != self.vectors[idx]:
            code = env.ROOT_MISMATCH if idx == 0 else env.WRONG_STATE
            return Issue(code, event.line, f"{event.node} 的向量与重放不符")
        self.frontier.remove(idx)
        self.current = idx
        self.state.checked_nodes += 1

        if self._terminal(self.vectors[idx]):
            self.phase = "need_goal"
            return None
        self.expected = self._expand_successors(idx)
        self.cursor = 0
        self.phase = "expanding"
        return None

    def _expand_successors(self, idx: int) -> list[tuple[str, int, str, Vector | None]]:
        """按规则数字 ID 升序计算应产生的 NEW/DUP，并即时登记新节点。"""
        out: list[tuple[str, int, str, Vector | None]] = []
        vec = self.vectors[idx]
        self.pending_new = []
        for rule in self.world.rules:
            child = (
                env.step(vec, rule)
                if self.forward
                else env.regress(vec, rule, self.world.total_units)
            )
            if child is None:
                continue
            known = self.visited.get(child)
            if known is not None:
                out.append(("DUP", known, rule.id, child))
                continue
            new_idx = len(self.vectors)
            self.vectors.append(child)
            self.parent.append(idx)
            self.via.append(rule.id)
            self.visited[child] = new_idx
            self.pending_new.append(new_idx)
            out.append(("NEW", new_idx, rule.id, child))
        return out

    def _on_new(self, event: Event) -> Issue | None:
        if self.phase != "expanding":
            return Issue(env.UNEXPECTED_EVENT, event.line, f"NEW 出现在 {self.phase}")
        if self.world.rule(event.rule or "") is None:
            return Issue(env.UNKNOWN_RULE, event.line, event.rule)
        return self._match_successor(event, "NEW")

    def _on_dup(self, event: Event) -> Issue | None:
        if self.phase != "expanding":
            return Issue(env.UNEXPECTED_EVENT, event.line, f"DUP 出现在 {self.phase}")
        if self.world.rule(event.rule or "") is None:
            return Issue(env.UNKNOWN_RULE, event.line, event.rule)
        return self._match_successor(event, "DUP")

    def _match_successor(self, event: Event, kind: str) -> Issue | None:
        if self.cursor >= len(self.expected):
            return Issue(env.EXTRA_SUCCESSOR, event.line, f"{event.rule} 不应产生后继")
        exp_kind, exp_idx, exp_rule, exp_vec = self.expected[self.cursor]
        if event.rule != exp_rule:
            later = [i for i in range(self.cursor + 1, len(self.expected)) if self.expected[i][2] == event.rule]
            if later:
                return Issue(env.SUCCESSOR_ORDER, event.line, f"期望 {exp_rule}，实际 {event.rule}")
            return Issue(env.EXTRA_SUCCESSOR, event.line, f"{event.rule} 不应产生后继")
        if kind != exp_kind:
            code = env.DUP_TARGET_MISMATCH if kind == "DUP" else env.EXTRA_SUCCESSOR
            return Issue(code, event.line, f"{event.rule} 期望 {exp_kind}，实际 {kind}")
        if kind == "NEW":
            if event.node_index != exp_idx:
                code = (
                    env.DUPLICATE_NODE_ID
                    if event.node_index < exp_idx
                    else env.BAD_NODE_ORDER
                )
                return Issue(code, event.line, f"期望 n{exp_idx}，实际 {event.node}")
            if event.vector != exp_vec:
                return Issue(env.WRONG_STATE, event.line, f"{event.node} 的向量与重放不符")
        else:
            if event.node_index != exp_idx:
                return Issue(
                    env.DUP_TARGET_MISMATCH, event.line, f"期望 n{exp_idx}，实际 {event.node}"
                )
        self.cursor += 1
        return None

    def _on_close(self, event: Event) -> Issue | None:
        if self.phase != "expanding":
            return Issue(env.UNEXPECTED_EVENT, event.line, f"CLOSE 出现在 {self.phase}")
        if event.node_index != self.current:
            return Issue(env.UNEXPECTED_EVENT, event.line, f"CLOSE 的节点不是当前展开节点")
        if self.cursor < len(self.expected):
            missing = self.expected[self.cursor]
            return Issue(env.MISSING_SUCCESSOR, event.line, f"漏掉 {missing[2]} 的 {missing[0]}")
        if self.schedule == "lifo":
            self.frontier.extend(reversed(self.pending_new))
        else:
            self.frontier.extend(self.pending_new)
        self.pending_new = []
        self.expected = []
        self.cursor = 0
        self.current = None
        self.phase = "need_expand"
        return None

    def _on_goal(self, event: Event) -> Issue | None:
        if self.phase == "expanding":
            return Issue(env.BAD_GOAL_CLAIM, event.line, "该节点不满足终止条件")
        if self.phase != "need_goal":
            return Issue(env.UNEXPECTED_EVENT, event.line, f"GOAL 出现在 {self.phase}")
        if event.node_index != self.current:
            return Issue(env.UNEXPECTED_EVENT, event.line, "GOAL 的节点不是当前展开节点")
        self.state.goal_node = event.node_index
        self.phase = "need_plan"
        return None

    def _on_plan(self, event: Event) -> Issue | None:
        if self.phase != "need_plan":
            return Issue(env.UNEXPECTED_EVENT, event.line, f"PLAN 出现在 {self.phase}")
        assert self.state.goal_node is not None
        expected = search._restore_plan(
            self.parent, self.via, self.state.goal_node, self.forward
        )
        observed = event.plan or []
        self.state.plan = observed
        self.state.plan_consistent = observed == expected
        # Any executable goal-reaching final plan is allowed. Keep agreement
        # with the search's parent chain as a diagnostic, not a replay failure.
        self.phase = "need_end"
        return None

    def _on_end(self, event: Event) -> Issue | None:
        if self.phase != "need_end":
            return Issue(env.UNEXPECTED_EVENT, event.line, f"END 出现在 {self.phase}")
        self.phase = "done"
        return None

    def _terminal(self, vec: Vector) -> bool:
        if self.forward:
            return env.meets_goal(vec, self.world.goal)
        return env.demand_met(vec, self.world.initial)


# --------------------------------------------------------------------------- #
# 验证返回值
# --------------------------------------------------------------------------- #


@dataclass
class VerifyResult:
    record_id: str
    world_id: str
    kind: str
    parse_status: str
    vector_format: str | None = None
    hack: bool | None = None
    hack_kind: str | None = None
    search_outcome: str | None = None
    frontier_remaining: int | None = None
    direction: str | None = None
    plan: list[str] | None = None
    plan_valid: bool | None = None
    plan_steps: int | None = None
    plan_shortest: int | None = None
    plan_waste: int | None = None
    first_goal_step: int | None = None
    steps_after_goal: int | None = None
    units_initial: int | None = None
    units_final: int | None = None
    units_consumed: int | None = None
    state_revisits: int | None = None
    units_monotone: bool | None = None
    search_valid: bool | None = None
    plan_trace_consistent: bool | None = None
    compatible_modes: list[str] | None = None
    mode_errors: dict[str, str] | None = None
    checked_nodes: int | None = None
    within_budget: bool | None = None
    first_divergence_line: int | None = None
    first_unique_line: int | None = None
    first_error_offset_to_divergence: int | None = None
    first_error_at_critical_step: bool | None = None
    first_unrecoverable_step: int | None = None
    first_error_frontier_size: int | None = None
    first_error: Issue | None = None
    text: str = ""

    def to_dict(self, *, include_text: bool = True) -> dict:
        payload = {
            "schema_version": env.SCHEMA_VERSION,
            "verification_policy": VERIFICATION_POLICY,
            "record_id": self.record_id,
            "world_id": self.world_id,
            "kind": self.kind,
            "parse_status": self.parse_status,
            "vector_format": self.vector_format,
            "hack": self.hack,
            "hack_kind": self.hack_kind,
            "search_outcome": self.search_outcome,
            "frontier_remaining": self.frontier_remaining,
            "direction": self.direction,
            "plan": self.plan,
            "plan_valid": self.plan_valid,
            "plan_steps": self.plan_steps,
            "plan_shortest": self.plan_shortest,
            "plan_waste": self.plan_waste,
            "first_goal_step": self.first_goal_step,
            "steps_after_goal": self.steps_after_goal,
            "units_initial": self.units_initial,
            "units_final": self.units_final,
            "units_consumed": self.units_consumed,
            "state_revisits": self.state_revisits,
            "units_monotone": self.units_monotone,
            "search_valid": self.search_valid,
            "plan_trace_consistent": self.plan_trace_consistent,
            "compatible_modes": self.compatible_modes,
            "mode_errors": self.mode_errors,
            "checked_nodes": self.checked_nodes,
            "within_budget": self.within_budget,
            "first_divergence_line": self.first_divergence_line,
            "first_unique_line": self.first_unique_line,
            "first_error_offset_to_divergence": self.first_error_offset_to_divergence,
            "first_error_at_critical_step": self.first_error_at_critical_step,
            "first_unrecoverable_step": self.first_unrecoverable_step,
            "first_error_frontier_size": self.first_error_frontier_size,
            "first_error": self.first_error.to_dict() if self.first_error else None,
        }
        if include_text:
            payload["text"] = self.text
        if hasattr(self, 'recheck_metrics'):
            payload['verification_policy']='search_recheck_v1'
            payload['recheck']=self.recheck_metrics
        return payload

    @property
    def passed(self) -> bool:
        """`--strict` 的通过条件，按 kind 区分可测量项。"""
        if self.parse_status == PARSE_ERROR_STATUS:
            return False
        if self.kind == KIND_PLAN:
            return bool(self.plan_valid)
        if self.kind == KIND_SEARCH:
            return bool(self.plan_valid and self.search_valid)
        return self.first_error is None


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def verify_record(
    world: World,
    text: str,
    *,
    kind: str,
    record_id: str = "",
    modes: Sequence[str] = search.MODES,
    budget: int | None = None,
    prefill: str | None = None,
    shared_lines: int | None = None,
    criticality: bool = False,
    criticality_states: int = env.DEFAULT_CRITICALITY_STATES,
    max_plan_steps: int | None = DEFAULT_MAX_PLAN_STEPS,
    max_events: int | None = DEFAULT_MAX_EVENTS,
    protocol: str = 'standard',
) -> VerifyResult:
    if protocol=='search_recheck_v1':
        if kind!='search':raise ValueError('recheck protocol requires search')
        from .recheck import verify as verify_recheck
        return verify_recheck(world,text,kind=kind,record_id=record_id,modes=modes,budget=budget,prefill=prefill,
            shared_lines=shared_lines,criticality=criticality,criticality_states=criticality_states,
            max_plan_steps=max_plan_steps,max_events=max_events)
    if protocol!='standard':raise ValueError('Unknown verification protocol')
    if kind not in KINDS:
        raise ValueError(f"未知 --kind: {kind}")
    raw = text
    text = restore_prefill(text, prefill)
    split = split_assistant_text(text)
    result = VerifyResult(
        record_id=record_id,
        world_id=world.world_id,
        kind=kind,
        parse_status=PARSE_COMPLETE,
        text=raw,
    )

    if max_events is not None and len(split.lines) > max_events:
        result.parse_status = PARSE_ERROR_STATUS
        result.first_error = Issue(
            env.TRACE_TOO_LONG, None, f"{len(split.lines)} 行，上限 {max_events}"
        )
        _annotate_divergence(result, shared_lines)
        return result

    if kind == KIND_SEARCH:
        # The final answer remains independently measurable even when an earlier
        # search event cannot be parsed. Do not guess a PLAN from prose or accept
        # ambiguous multiple PLAN lines. The event-count protection above still
        # takes precedence over all parsing/execution.
        result.search_valid = False
        result.search_outcome = OUTCOME_INVALID
        result.plan_valid = False
        plan_lines = [(i, line) for i, line in enumerate(split.lines, 1)
                      if line == "PLAN" or line.startswith("PLAN ")]
        if len(plan_lines) == 1:
            lineno, line = plan_lines[0]
            plan_event = parse_event(line, lineno, world.n_resources, world.symbols)
            if isinstance(plan_event, Event):
                result.plan = plan_event.plan or []
                _apply_plan_check(result, env.verify_plan(world, result.plan, max_steps=max_plan_steps))
        _classify_hack(result)

    allow_partial_tail = kind == KIND_PREFIX and not text.endswith("\n")
    events: list[Event] = []
    pending_tail = False
    for lineno, line in enumerate(split.lines, start=1):
        parsed = parse_event(line, lineno, world.n_resources, world.symbols)
        if isinstance(parsed, Issue):
            is_tail = lineno == len(split.lines)
            if allow_partial_tail and is_tail and is_partial_event(line, world.n_resources):
                pending_tail = True
                break
            result.parse_status = PARSE_ERROR_STATUS
            result.first_error = parsed
            _annotate_divergence(result, shared_lines)
            return result
        events.append(parsed)

    if kind == KIND_SEARCH and split.had_think and not split.think_closed:
        result.parse_status = PARSE_PREFIX
        result.first_error = Issue(env.INCOMPLETE_TRACE, len(split.lines), "think 块缺少 </think>")
        _classify_hack(result)
        _annotate_divergence(result, shared_lines)
        return result

    if split.had_think and split.think_closed and split.plan_inside_think:
        result.parse_status = PARSE_ERROR_STATUS
        result.first_error = Issue(env.UNEXPECTED_EVENT, 1, "PLAN/END 出现在 think 块内")
        _annotate_divergence(result, shared_lines)
        return result

    if kind == KIND_PLAN:
        _verify_plan_only(world, events, result, max_plan_steps)
        if criticality:
            _annotate_plan_criticality(world, result, criticality_states)
        _annotate_divergence(result, shared_lines)
        return result

    formats = {e.vector_format for e in events if e.vector_format is not None}
    if len(formats) > 1:
        result.parse_status = PARSE_ERROR_STATUS
        result.first_error = Issue(
            env.VECTOR_FORMAT_MIXED, 1, "轨迹内混用 dense 与 sparse 编码"
        )
        _annotate_divergence(result, shared_lines)
        return result
    result.vector_format = next(iter(formats)) if formats else None

    directions = {e.keyword for e in events if e.keyword is not None}
    if len(directions) > 1:
        result.parse_status = PARSE_ERROR_STATUS
        result.first_error = Issue(env.DIRECTION_MISMATCH, 1, "轨迹内混用 STATE 与 NEED")
        _annotate_divergence(result, shared_lines)
        return result
    keyword = next(iter(directions)) if directions else None
    direction = None
    if keyword == "STATE":
        direction = "forward"
    elif keyword == "NEED":
        direction = "backward"
    result.direction = direction

    has_end = any(e.kind == "END" for e in events)
    if pending_tail:
        result.parse_status = PARSE_PENDING
    elif not has_end:
        result.parse_status = PARSE_PREFIX

    candidates = [m for m in modes]
    generic_direction = direction or "forward"
    generic = Replay(world, generic_direction, "any")
    generic_state = generic.run(events)
    result.checked_nodes = generic_state.checked_nodes

    fail_line: dict[str, float] = {}
    mode_errors: dict[str, str] = {}
    for mode in candidates:
        if direction is not None and search.direction_of(mode) != direction:
            fail_line[mode] = 1.0
            mode_errors[mode] = env.DIRECTION_MISMATCH
            continue
        replay = Replay(world, search.direction_of(mode), search.schedule_of(mode))
        state = replay.run(events)
        if state.error is None:
            fail_line[mode] = math.inf
        else:
            fail_line[mode] = float(state.error.line or 0)
            mode_errors[mode] = state.error.code

    compatible = [m for m in candidates if fail_line.get(m) == math.inf]
    result.compatible_modes = compatible
    result.mode_errors = mode_errors or None
    finite = [v for v in fail_line.values() if v != math.inf]
    result.first_divergence_line = int(min(finite)) if finite else None
    if len(compatible) == 1 and len(candidates) > 1 and len(finite) == len(candidates) - 1:
        result.first_unique_line = int(max(finite))
    elif len(candidates) == 1 and compatible:
        result.first_unique_line = 1

    if kind == KIND_PREFIX:
        result.first_error = generic_state.error
        result.plan_valid = None
        result.search_valid = None
        result.plan_trace_consistent = None
        if budget is not None:
            result.within_budget = result.checked_nodes <= budget
        if criticality:
            result.first_error_frontier_size = generic_state.frontier_at_error
        result.frontier_remaining = generic_state.frontier_remaining
        result.search_outcome = _classify_search(
            generic_state, has_end=has_end, budget=budget, plan_valid=None
        )
        _annotate_divergence(result, shared_lines)
        return result

    # kind == search
    if generic_state.error is not None:
        result.search_valid = False
        result.first_error = generic_state.error
    elif result.parse_status in (PARSE_PREFIX, PARSE_PENDING):
        result.search_valid = False
        result.first_error = Issue(env.INCOMPLETE_TRACE, len(split.lines), "轨迹未到 END")
    else:
        result.search_valid = True

    result.plan_trace_consistent = generic_state.plan_consistent

    # 第 4.3 节：最终方案的有效性独立于搜索过程是否正确。即使重放在更早的位置报错，
    # 只要轨迹里有一条合法的 PLAN 行，仍然按它给出 plan_valid。
    plan_events = [e for e in events if e.kind == "PLAN"]
    if len(plan_events) == 1:
        plan = plan_events[0].plan or []
        result.plan = plan
        check = env.verify_plan(world, plan, max_steps=max_plan_steps)
        _apply_plan_check(result, check)
        if result.first_error is None and not check.valid:
            result.first_error = check.error
    else:
        result.plan = None
        result.plan_valid = False
        if result.first_error is None:
            detail = "缺少 PLAN" if not plan_events else f"PLAN 行数为 {len(plan_events)}"
            result.first_error = Issue(env.INCOMPLETE_TRACE, len(split.lines), detail)
    if budget is not None:
        result.within_budget = result.checked_nodes <= budget
    result.frontier_remaining = generic_state.frontier_remaining
    result.search_outcome = _classify_search(
        generic_state, has_end=has_end, budget=budget, plan_valid=result.plan_valid
    )
    _classify_hack(result)
    if criticality:
        result.first_error_frontier_size = generic_state.frontier_at_error
        if result.plan is not None:
            _annotate_plan_criticality(world, result, criticality_states)
    _annotate_divergence(result, shared_lines)
    return result


HACK_PROCESS_ERROR = "process_error"
HACK_PLAN_TRACE_MISMATCH = "plan_trace_mismatch"
HACK_INCOMPLETE_SEARCH = "incomplete_search"


def _classify_hack(result: VerifyResult) -> None:
    """最终答案有效而搜索过程无效：标记为 hack，联合成功指标计为失败。

    包括过程错误和未完成的过程。有效的替代路径或非最优路径本身不属于 hack。
    环境只标记数据属性，不推断模型意图。
    """
    if result.plan_valid is not True or result.search_valid is not False:
        result.hack = False if result.plan_valid is not None else None
        result.hack_kind = None
        return
    result.hack = True
    code = result.first_error.code if result.first_error else None
    if code == env.PLAN_TRACE_MISMATCH:
        result.hack_kind = HACK_PLAN_TRACE_MISMATCH
    elif code == env.INCOMPLETE_TRACE:
        result.hack_kind = HACK_INCOMPLETE_SEARCH
    else:
        result.hack_kind = HACK_PROCESS_ERROR


def _classify_search(
    state: ReplayState, *, has_end: bool, budget: int | None, plan_valid: bool | None
) -> str:
    """把一条搜索输出归入五类，区分「过程错误」与「合法但未找到解」。

    `budget_exhausted` 与 `stopped_early` 的差别在于模型是否真的用满了预算：前者是
    预算约束下的正确行为，后者是提前放弃。
    """
    if state.error is not None:
        return OUTCOME_INVALID
    if state.goal_node is not None:
        if has_end and plan_valid is not False:
            return OUTCOME_SOLVED
        return OUTCOME_INVALID if plan_valid is False else OUTCOME_STOPPED_EARLY
    if state.frontier_remaining == 0 and state.checked_nodes > 0:
        # 合法地把 frontier 取空而未达目标。生成器保证题目可解，因此这一结果与可解性
        # 矛盾，出现时应作为实现异常调查。
        return OUTCOME_FRONTIER_EMPTY
    if budget is not None and state.checked_nodes >= budget:
        return OUTCOME_BUDGET_EXHAUSTED
    return OUTCOME_STOPPED_EARLY


def _annotate_divergence(result: VerifyResult, shared_lines: int | None) -> None:
    """首错行号相对该题分歧位置的偏移（第 8.4 节）。"""
    if shared_lines is None or result.first_error is None or result.first_error.line is None:
        return
    result.first_error_offset_to_divergence = result.first_error.line - shared_lines


def _annotate_plan_criticality(
    world: World, result: VerifyResult, max_states: int
) -> None:
    """按模型自身的方案前缀判定首错步是否为关键步，并找出首个不可恢复的步。

    两项都只用于报告。首错步的类别按该步起点状态上的全部可执行规则计算，与模型选的
    规则是否可执行无关。
    """
    plan = result.plan or []
    status, distance = env.shortest_distance(world, world.initial, max_states)
    if status == env.REACHABLE and distance is not None:
        result.plan_shortest = distance
        if result.plan_valid and result.first_goal_step is not None:
            result.plan_waste = result.first_goal_step - distance
    state = world.initial
    for t, rule_id in enumerate(plan, start=1):
        if (
            result.first_error_at_critical_step is None
            and result.first_error is not None
            and result.first_error.line == t
        ):
            outcome = env.step_outcome(world, state, rule_id, plan[t:], max_states)
            result.first_error_at_critical_step = outcome.kind == env.KIND_CRITICAL
        rule = world.rule(rule_id)
        if rule is None:
            break
        nxt = env.step(state, rule)
        if nxt is None:
            break
        state = nxt
        if result.first_unrecoverable_step is None:
            verdict = env.goal_reachable(world, state, max_states)
            if verdict is False:
                result.first_unrecoverable_step = t


def _apply_plan_check(result: VerifyResult, check: env.PlanCheck) -> None:
    """把方案验证的资源记账写入返回值。"""
    result.plan_valid = check.valid
    result.plan_steps = check.steps
    result.first_goal_step = check.first_goal_step
    result.steps_after_goal = check.steps_after_goal
    result.units_initial = check.units_initial
    result.units_final = check.units_final
    result.units_consumed = check.units_consumed
    result.state_revisits = check.state_revisits
    result.units_monotone = check.units_monotone


def _verify_plan_only(
    world: World,
    events: Sequence[Event],
    result: VerifyResult,
    max_plan_steps: int | None = DEFAULT_MAX_PLAN_STEPS,
) -> VerifyResult:
    """kind=plan：只验证最终方案，不输出伪模式分类。"""
    plan_events = [e for e in events if e.kind == "PLAN"]
    if len(plan_events) != 1:
        result.parse_status = PARSE_ERROR_STATUS
        result.first_error = Issue(env.PARSE_ERROR, None, f"PLAN 行数为 {len(plan_events)}")
        return result
    if not events or events[-1].kind != "END":
        result.parse_status = PARSE_ERROR_STATUS
        result.first_error = Issue(env.PARSE_ERROR, None, "末行不是 END")
        return result
    plan = plan_events[0].plan or []
    check = env.verify_plan(world, plan, max_steps=max_plan_steps)
    result.plan = plan
    _apply_plan_check(result, check)
    result.first_error = check.error
    result.search_valid = None
    result.plan_trace_consistent = None
    result.compatible_modes = None
    result.mode_errors = None
    result.checked_nodes = None
    result.within_budget = None
    result.direction = None
    return result


def verify_search(
    world: World,
    events: Sequence[str],
    modes: Sequence[str] = search.MODES,
    budget: int | None = None,
    prefix: bool = False,
) -> VerifyResult:
    """第 12.1 节的内部接口：直接接收事件行列表。"""
    text = "\n".join(events)
    if not prefix:
        text += "\n"
    return verify_record(
        world,
        text,
        kind=KIND_PREFIX if prefix else KIND_SEARCH,
        modes=modes,
        budget=budget,
    )


# --------------------------------------------------------------------------- #
# 预测输入的 schema 识别
# --------------------------------------------------------------------------- #


def coerce_prediction(payload: dict, index: int) -> tuple[str, str, str]:
    """识别预测输入记录，返回 (record_id, world_id, text)。

    支持 text、合并 QA（model_answer）、solutions（trace）、export（messages）。
    不从任意段落中猜测答案。
    """
    if not isinstance(payload, dict):
        raise env.WorldError(f"第 {index} 条记录必须是 JSON 对象")
    world_id = payload.get("world_id")
    if not isinstance(world_id, str) or not world_id:
        raise env.WorldError(f"第 {index} 条记录缺少 world_id")
    if "model_answer" in payload:
        if not isinstance(payload["model_answer"], str):
            raise env.WorldError(f"第 {index} 条记录的 model_answer 必须是字符串")
        if "text" in payload and payload["text"] != payload["model_answer"]:
            raise env.WorldError(f"第 {index} 条记录的 text 与 model_answer 不一致")
        return str(payload.get("record_id") or f"pred_{index:06d}"), world_id, payload["model_answer"]
    if "text" in payload:
        if not isinstance(payload["text"], str):
            raise env.WorldError(f"第 {index} 条记录的 text 必须是字符串")
        record_id = str(payload.get("record_id") or f"pred_{index:06d}")
        return record_id, str(world_id), str(payload["text"])
    if "trace" in payload and payload.get("trace") is not None:
        if not isinstance(payload["trace"], list) or not all(isinstance(line, str) for line in payload["trace"]):
            raise env.WorldError(f"第 {index} 条记录的 trace 必须是字符串列表")
        mode = payload.get("mode", "")
        record_id = str(payload.get("record_id") or f"{world_id}:{mode}")
        return record_id, str(world_id), "\n".join(payload["trace"]) + "\n"
    if "messages" in payload:
        if not isinstance(payload["messages"], list) or not all(isinstance(msg, dict) for msg in payload["messages"]):
            raise env.WorldError(f"第 {index} 条记录的 messages 必须是消息对象列表")
        content = None
        for message in payload["messages"]:
            if message.get("role") == "assistant":
                content = message.get("content")
        if not isinstance(content, str):
            raise env.WorldError(f"第 {index} 条记录的 messages 缺少 assistant")
        record_id = str(payload.get("record_id") or f"pred_{index:06d}")
        return record_id, str(world_id), str(content)
    raise env.WorldError(f"第 {index} 条记录的 schema 无法识别")

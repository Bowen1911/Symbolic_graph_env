"""Qwen3 QA 渲染：Q 文本模板（附录 A）、think/非 think 两种视图、target_hash。

本模块不依赖 tokenizer；`measure_tokens` 仅在显式传入 tokenizer 时使用。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

from . import env, search
from .env import World
from .search import SolveResult

VIEW_PLAN = "plan"
VIEW_SEARCH = "search"
VIEWS = (VIEW_PLAN, VIEW_SEARCH)
THINKING_OFF = "off"
THINKING_ON = "on"
THINKING = (THINKING_OFF, THINKING_ON)

# 首版只允许这两种组合（第 8.3 节）。
ALLOWED_PROTOCOLS = {(VIEW_PLAN, THINKING_OFF), (VIEW_SEARCH, THINKING_ON)}

PROMPT_MODE_NONE = "none"
PROMPT_MODE_TEACHER = "teacher"
PROMPT_MODES = (PROMPT_MODE_NONE, PROMPT_MODE_TEACHER)

PROMPT_DETAILED = "detailed"
PROMPT_STRUCTURED = "structured"
PROMPT_STYLES = (PROMPT_STRUCTURED, PROMPT_DETAILED)

NONTHINK_PREFIX = "<think>\n\n</think>\n\n"
THINK_PREFIX = "<think>\n"
IM_START_USER = "<|im_start|>user\n"
IM_END = "<|im_end|>"
IM_START_ASSISTANT = "<|im_start|>assistant\n"

TASK_TEXT = """A state gives a count for each resource. Rule r applies in state s when s has at
least the input count of every resource of r. Applying r subtracts its inputs and
adds its outputs. Resources are consumed; nothing is reused. The goal is reached
when every resource count is at least its GOAL count; extra resources are allowed.

Find a rule sequence that is executable from INITIAL and reaches GOAL. The
sequence does not have to be the shortest."""

OUTPUT_PLAN = """Answer with:
PLAN <rule ids separated by single spaces>
END"""

COUNTS_DENSE = """<counts> gives one integer per resource in RESOURCES order, separated by single
spaces."""

COUNTS_SPARSE = """<counts> lists only the resources whose count is not zero, each written as
<resource>:<count>, in RESOURCES order and separated by single spaces. A state
in which every count is zero is written as a single -."""

OUTPUT_SEARCH_FORWARD = """Record the search, then give the plan. Write one event per line:
EXPAND <node> STATE <counts>            take the next node from the frontier and check it
NEW <node> VIA <rule> STATE <counts>    a rule applied to the expanded node gives a state seen for the first time
DUP <node> VIA <rule>                   a rule applied to the expanded node gives the state of existing node <node>
CLOSE <node>                            every rule has been enumerated for the expanded node
GOAL <node>                             the expanded node reaches GOAL
{COUNTS}
The root node is n0 with the INITIAL counts. Nodes are numbered n1, n2,
... in order of first discovery. For each expanded node, enumerate rules in
increasing rule id; a rule that does not apply produces no line.

After GOAL, answer with:
PLAN <rule ids separated by single spaces>
END"""

OUTPUT_SEARCH_BACKWARD = """Record the search, then give the plan. The search runs backward from GOAL over
demand vectors. Regressing demand d through rule r gives, componentwise,
inputs(r) + max(d - outputs(r), 0). Write one event per line:
EXPAND <node> NEED <counts>             take the next node from the frontier and check it
NEW <node> VIA <rule> NEED <counts>     regressing the expanded node through a rule gives a demand seen for the first time
DUP <node> VIA <rule>                   regressing the expanded node through a rule gives the demand of existing node <node>
CLOSE <node>                            every rule has been enumerated for the expanded node
GOAL <node>                             every demand count of the expanded node is met by INITIAL
{COUNTS}
The root node is n0 with the GOAL counts. Nodes are numbered n1, n2, ...
in order of first discovery. For each expanded node, enumerate rules in
increasing rule id; a regression whose total demand exceeds the total count of
INITIAL produces no line.

After GOAL, reverse the rule sequence from n0 to that node and answer with:
PLAN <rule ids separated by single spaces>
END"""

SCHEDULE_FIFO_LINE = "Expand nodes in first-in-first-out order."
SCHEDULE_LIFO_LINE = (
    "Expand nodes in last-in-first-out order; among the successors of one node, "
    "the one discovered first is expanded first."
)
BUDGET_LINE = "Reach GOAL within SEARCH_BUDGET EXPAND events."


def check_protocol(view: str, thinking: str) -> None:
    if view not in VIEWS:
        raise ValueError(f"未知 --view: {view}")
    if thinking not in THINKING:
        raise ValueError(f"未知 --thinking: {thinking}")
    if (view, thinking) not in ALLOWED_PROTOCOLS:
        raise ValueError(
            f"首版不支持 --view {view} --thinking {thinking}；"
            "合法组合为 plan/off 与 search/on"
        )


def _counts_text(vector_format: str) -> str:
    if vector_format == env.VECTOR_DENSE:
        return COUNTS_DENSE
    if vector_format == env.VECTOR_SPARSE:
        return COUNTS_SPARSE
    raise ValueError(f"未知向量编码: {vector_format}")


def prompt_variant(
    view: str, directions: Sequence[str], vector_format: str = env.VECTOR_DENSE,
    prompt_style: str = PROMPT_DETAILED,
) -> str:
    """模板变体标识，写入 manifest 并参与 target_hash（第 10.2 节）。"""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"未知 prompt_style: {prompt_style}")
    if prompt_style == PROMPT_STRUCTURED:
        return f"{view}-structured-{vector_format}-v2"
    if view == VIEW_PLAN:
        return "plan"
    uniq = tuple(dict.fromkeys(directions))
    base = (
        "search-forward"
        if uniq == ("forward",)
        else "search-backward"
        if uniq == ("backward",)
        else "search-both"
    )
    return base if vector_format == env.VECTOR_DENSE else f"{base}-{vector_format}"


# --------------------------------------------------------------------------- #
# Q 渲染
# --------------------------------------------------------------------------- #


def _nonzero_terms(symbols: Sequence[str], vec: Sequence[int]) -> str:
    return " ".join(f"{symbols[i]}:{v}" for i, v in enumerate(vec) if v)


def render_header(
    world: World, *, budget: int | None = None, teacher_mode: str | None = None
) -> str:
    lines = [
        "RESOURCES " + " ".join(world.symbols),
        "INITIAL " + _nonzero_terms(world.symbols, world.initial),
        "GOAL " + _nonzero_terms(world.symbols, world.goal),
        "RULES",
    ]
    for rule in world.rules:
        lines.append(
            f"{rule.id}: {_nonzero_terms(world.symbols, rule.consume)}"
            f" -> {_nonzero_terms(world.symbols, rule.produce)}"
        )
    if budget is not None:
        lines.append(f"SEARCH_BUDGET {budget}")
    if teacher_mode is not None:
        lines.append(f"SEARCH_MODE {teacher_mode}")
    return "\n".join(lines)


def render_question(
    world: World,
    *,
    view: str,
    directions: Sequence[str] = ("forward",),
    budget: int | None = None,
    teacher_mode: str | None = None,
    vector_format: str = env.VECTOR_DENSE,
    prompt_style: str = PROMPT_DETAILED,
) -> str:
    """按附录 A 组装 Q。事件语法段只覆盖 directions 给出的方向集合。"""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"未知 prompt_style: {prompt_style}")
    if prompt_style == PROMPT_STRUCTURED:
        return render_header(world, budget=budget, teacher_mode=teacher_mode)
    counts = _counts_text(vector_format)
    blocks = [render_header(world, budget=budget, teacher_mode=teacher_mode), TASK_TEXT]

    extra: list[str] = []
    if teacher_mode is not None:
        schedule = search.schedule_of(teacher_mode)
        extra.append(SCHEDULE_FIFO_LINE if schedule == "fifo" else SCHEDULE_LIFO_LINE)
    if budget is not None:
        extra.append(BUDGET_LINE)
    if extra:
        blocks.append("\n".join(extra))

    if view == VIEW_PLAN:
        blocks.append(OUTPUT_PLAN)
    else:
        uniq = list(dict.fromkeys(directions))
        if "forward" in uniq:
            blocks.append(OUTPUT_SEARCH_FORWARD.replace("{COUNTS}", counts))
        if "backward" in uniq:
            blocks.append(OUTPUT_SEARCH_BACKWARD.replace("{COUNTS}", counts))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# assistant 渲染
# --------------------------------------------------------------------------- #


def split_trace(trace: Sequence[str]) -> tuple[list[str], str, str]:
    """把完整轨迹拆成 (搜索事件, PLAN 行, END 行)。"""
    plan_idx = None
    for i, line in enumerate(trace):
        if line == "PLAN" or line.startswith("PLAN "):
            plan_idx = i
            break
    if plan_idx is None or plan_idx + 1 >= len(trace) or trace[plan_idx + 1] != "END":
        raise ValueError("轨迹缺少 PLAN/END，不能渲染为成功 QA")
    return list(trace[:plan_idx]), trace[plan_idx], trace[plan_idx + 1]


def render_assistant(result: SolveResult, view: str, thinking: str) -> str:
    """同一 SolveResult 渲染两种视图；PLAN/END 一致（第 10.1 节）。"""
    check_protocol(view, thinking)
    if not result.solved:
        raise ValueError(f"{result.world_id}:{result.mode} 状态为 {result.status}，不可导出")
    events, plan_line, end_line = split_trace(result.trace)
    tail = f"{plan_line}\n{end_line}"
    if view == VIEW_PLAN:
        return f"{NONTHINK_PREFIX}{tail}"
    body = "\n".join(events)
    return f"{THINK_PREFIX}{body}\n</think>\n\n{tail}"


def assistant_prefill(view: str) -> str:
    return NONTHINK_PREFIX if view == VIEW_PLAN else THINK_PREFIX


def render_train_text(question: str, assistant: str) -> str:
    return IM_START_USER + question + IM_END + "\n" + IM_START_ASSISTANT + assistant + IM_END


def render_inference_text(question: str, view: str) -> str:
    return IM_START_USER + question + IM_END + "\n" + IM_START_ASSISTANT + assistant_prefill(view)


# --------------------------------------------------------------------------- #
# target_hash
# --------------------------------------------------------------------------- #


def _canonical_segment(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def target_hash(world_id: str, question: str, assistant: str) -> str:
    """第 10.2 节：三段规范化后以 \\x00 连接，取 SHA-256 前 16 字节。"""
    payload = "\x00".join(
        [_canonical_segment(world_id), _canonical_segment(question), _canonical_segment(assistant)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# QA 记录
# --------------------------------------------------------------------------- #


@dataclass
class QARecord:
    record_id: str
    world_id: str
    group_id: str
    structure: str
    source_mode: str
    view: str
    thinking: str
    prompt_variant: str
    question: str
    assistant: str
    assistant_prefill: str
    target_hash: str
    target_weight: float = 0.0
    plan_length: int = 0
    trace_events: int = 0
    checked_nodes: int = 0
    char_lengths: dict = field(default_factory=dict)
    token_lengths: dict | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": env.SCHEMA_VERSION,
            "record_id": self.record_id,
            "world_id": self.world_id,
            "group_id": self.group_id,
            "structure": self.structure,
            "source_mode": self.source_mode,
            "view": self.view,
            "thinking": self.thinking,
            "prompt_variant": self.prompt_variant,
            "messages": [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.assistant},
            ],
            "assistant_prefill": self.assistant_prefill,
            "target_hash": self.target_hash,
            "target_weight": self.target_weight,
            "plan_length": self.plan_length,
            "trace_events": self.trace_events,
            "checked_nodes": self.checked_nodes,
            "char_lengths": self.char_lengths,
            "token_lengths": self.token_lengths,
        }


def render_qa(
    world: World,
    result: SolveResult,
    *,
    view: str,
    thinking: str,
    directions: Sequence[str] = ("forward",),
    budget: int | None = None,
    prompt_mode: str = PROMPT_MODE_NONE,
    prompt_style: str = PROMPT_DETAILED,
) -> QARecord:
    check_protocol(view, thinking)
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"未知 --prompt-mode: {prompt_mode}")
    vector_format = result.limits.vector_format
    teacher_mode = result.mode if prompt_mode == PROMPT_MODE_TEACHER else None
    if teacher_mode is not None:
        # 显式指定模式时 Q 已给出方向，事件语法段只描述该方向。
        directions = (search.direction_of(teacher_mode),)
    question = render_question(
        world,
        view=view,
        directions=directions,
        budget=budget,
        teacher_mode=teacher_mode,
        vector_format=vector_format,
        prompt_style=prompt_style,
    )
    assistant = render_assistant(result, view, thinking)
    events, _, _ = split_trace(result.trace)
    return QARecord(
        record_id=f"{world.world_id}:{result.mode}:{view}",
        world_id=world.world_id,
        group_id=world.group_id,
        structure=str(world.meta.get("structure", "")),
        source_mode=result.mode,
        view=view,
        thinking=thinking,
        prompt_variant=prompt_variant(view, directions, vector_format, prompt_style),
        question=question,
        assistant=assistant,
        assistant_prefill=assistant_prefill(view),
        target_hash=target_hash(world.world_id, question, assistant),
        plan_length=len(result.plan or []),
        trace_events=len(events),
        checked_nodes=result.cost.checked_nodes,
        char_lengths={
            "question": len(question),
            "assistant": len(assistant),
            "think": len("\n".join(events)) if view == VIEW_SEARCH else 0,
        },
    )


# --------------------------------------------------------------------------- #
# 长度审计
# --------------------------------------------------------------------------- #


def load_tokenizer(path: str):
    from transformers import AutoTokenizer  # 延迟导入，首版核心路径不依赖

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def measure_tokens(tokenizer, record: QARecord) -> dict:
    """按实际渲染文本计量，分别报告输入、think、最终方案 token 数。"""

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    question_text = IM_START_USER + record.question + IM_END + "\n" + IM_START_ASSISTANT
    prefix = record.assistant_prefill
    if record.view == VIEW_SEARCH:
        think_text = record.assistant[len(prefix) :].split("</think>")[0]
        tail = record.assistant.split("</think>", 1)[1]
    else:
        think_text = ""
        tail = record.assistant[len(prefix) :]
    n_input = count(question_text)
    n_think = count(prefix + think_text + "</think>") if record.view == VIEW_SEARCH else count(prefix)
    n_plan = count(tail + IM_END)
    return {
        "input": n_input,
        "think": n_think,
        "plan": n_plan,
        "assistant": n_think + n_plan,
        "total": n_input + n_think + n_plan,
    }

"""错误输出的判定：覆盖模型可能产生的各类错误形态（第 4.3、8.4 节）。

分四组：最终方案类错误、搜索过程类错误、过程与答案的四种组合、有限资源内未找到解。
"""

from __future__ import annotations

import pytest

from symplan import env, search, verify
from symplan.env import Rule, World
from tests.fixtures import binary_tree_world, example_world


@pytest.fixture
def tree():
    return binary_tree_world(3)


@pytest.fixture
def body(tree):
    """正确轨迹去掉末尾的 PLAN/END，便于拼接各种结尾。"""
    trace = search.solve(tree, "forward-bfs").trace
    return [ln for ln in trace if not (ln.startswith("PLAN") or ln == "END")]


def _text(lines) -> str:
    return "\n".join(lines) + "\n"


# --- 一、最终方案类错误 ----------------------------------------------------- #


@pytest.mark.parametrize(
    "plan, code",
    [
        ("r0 r99", env.UNKNOWN_RULE),
        ("r2 r0", env.INSUFFICIENT_RESOURCE),
        ("r0", env.GOAL_NOT_MET),
        ("", env.GOAL_NOT_MET),
    ],
)
def test_plan_errors_are_detected(tree, plan, code):
    text = f"PLAN{(' ' + plan) if plan else ''}\nEND\n"
    result = verify.verify_record(tree, text, kind="plan")
    assert result.plan_valid is False
    assert result.first_error.code == code
    assert not result.passed


@pytest.mark.parametrize(
    "text",
    [
        "I think the answer is r0 then r2.\n",
        "PLAN r0 r2 r6\n",
        "PLAN r0\nPLAN r2\nEND\n",
        "END\n",
        "",
    ],
)
def test_malformed_plan_output_is_parse_error(tree, text):
    result = verify.verify_record(tree, text, kind="plan")
    assert result.parse_status == verify.PARSE_ERROR_STATUS
    assert result.first_error is not None
    assert not result.passed


# --- 二、搜索过程类错误 ----------------------------------------------------- #


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda b: [ln.replace("STATE 0 1 0", "STATE 0 9 0") for ln in b], env.WRONG_STATE),
        (lambda b: [ln for ln in b if not ln.startswith("NEW n2 ")], env.MISSING_SUCCESSOR),
        (lambda b: [ln.replace("NEW n1 ", "NEW n7 ") for ln in b], env.BAD_NODE_ORDER),
        (lambda b: [ln.replace("CLOSE n0", "GOAL n0") for ln in b], env.BAD_GOAL_CLAIM),
        (lambda b: [ln.replace("STATE", "NEED") for ln in b], env.ROOT_MISMATCH),
        (lambda b: [ln for ln in b if ln != "CLOSE n0"], env.UNEXPECTED_EVENT),
    ],
)
def test_search_process_errors_are_detected(tree, body, mutate, code):
    result = verify.verify_record(
        tree, _text(mutate(list(body))), kind="search", modes=("forward-bfs",)
    )
    assert result.search_valid is False
    assert result.search_outcome == verify.OUTCOME_INVALID
    assert result.first_error.code == code


def test_expanding_an_unknown_node_is_detected(tree, body):
    mutated = [
        ln.replace("EXPAND n1 ", "EXPAND n99 ") if ln.startswith("EXPAND n1 ") else ln
        for ln in body
    ]
    result = verify.verify_record(tree, _text(mutated), kind="search", modes=("forward-bfs",))
    assert result.search_valid is False
    assert result.first_error.code == env.UNKNOWN_NODE


def test_bad_frontier_choice_is_detected(tree, body):
    """取了一个已经被取出的节点。"""
    idx = next(i for i, ln in enumerate(body) if ln.startswith("EXPAND n1 "))
    mutated = body[:idx] + [body[0]] + body[idx:]
    result = verify.verify_record(tree, _text(mutated), kind="search", modes=("forward-bfs",))
    assert result.search_valid is False
    assert result.first_error.code == env.BAD_FRONTIER_CHOICE


# --- 三、过程与答案的四种组合（第 4.3 节） --------------------------------- #


def test_four_combinations_of_process_and_answer(tree, body):
    good_plan = ["PLAN r0 r2 r6", "END"]
    bad_plan = ["PLAN r0 r0", "END"]
    broken = [ln for ln in body if not ln.startswith("NEW n2 ")]

    ok_ok = verify.verify_record(tree, _text(body + good_plan), kind="search")
    assert (ok_ok.plan_valid, ok_ok.search_valid) == (True, True)
    assert ok_ok.search_outcome == verify.OUTCOME_SOLVED
    assert ok_ok.plan_trace_consistent is True

    # 过程错、答案对：第 4.3 节要求 plan_valid 仍为 true
    bad_ok = verify.verify_record(tree, _text(broken + good_plan), kind="search")
    assert bad_ok.plan_valid is True
    assert bad_ok.search_valid is False
    assert bad_ok.search_outcome == verify.OUTCOME_INVALID
    assert bad_ok.first_error.code == env.MISSING_SUCCESSOR

    ok_bad = verify.verify_record(tree, _text(body + bad_plan), kind="search")
    assert ok_bad.plan_valid is False
    assert ok_bad.search_valid is True
    assert not ok_bad.passed

    bad_bad = verify.verify_record(tree, _text(broken + bad_plan), kind="search")
    assert bad_bad.plan_valid is False
    assert bad_bad.search_valid is False


def test_jumping_to_plan_keeps_answer_validity(tree, body):
    """中途放弃搜索直接给出有效方案：答案有效，过程无效。"""
    result = verify.verify_record(
        tree, _text(body[:3] + ["PLAN r0 r2 r6", "END"]), kind="search"
    )
    assert result.plan_valid is True
    assert result.search_valid is False
    assert result.first_error.code == env.UNEXPECTED_EVENT


# --- 四、有限资源内未找到解 ------------------------------------------------- #


def test_budget_exhausted_is_distinguished_from_process_error(tree):
    partial = search.solve(tree, "forward-bfs", search.Limits(max_expansions=5))
    assert partial.status == search.STATUS_BUDGET
    text = _text(partial.trace)

    as_prefix = verify.verify_record(tree, text, kind="prefix", budget=5)
    assert as_prefix.first_error is None
    assert as_prefix.search_outcome == verify.OUTCOME_BUDGET_EXHAUSTED
    assert as_prefix.checked_nodes == 5

    # --kind search 要求完整轨迹，因此 search_valid 为 false，但原因由 outcome 区分
    as_search = verify.verify_record(tree, text, kind="search", budget=5)
    assert as_search.search_valid is False
    assert as_search.first_error.code == env.INCOMPLETE_TRACE
    assert as_search.search_outcome == verify.OUTCOME_BUDGET_EXHAUSTED


def test_stopping_early_is_distinguished_from_budget_exhausted(tree, body):
    """只检查了 2 个节点就停止，预算是 5：属于提前放弃。"""
    short = [ln for ln in body[:7]]
    result = verify.verify_record(tree, _text(short), kind="prefix", budget=5)
    assert result.first_error is None
    assert result.search_outcome == verify.OUTCOME_STOPPED_EARLY
    assert result.checked_nodes < 5


def test_frontier_emptied_without_goal_is_flagged():
    """合法地把 frontier 取空而未达目标。可解问题上不应出现。"""
    world = World(
        symbols=("A", "B", "G"),
        initial=(1, 0, 0),
        goal=(0, 0, 1),
        rules=(Rule("r0", (1, 0, 0), (0, 1, 0)),),
        world_id="w_unsolvable",
        group_id="g_unsolvable",
    )
    env.validate_world(world)
    result = search.solve(world, "forward-bfs")
    assert result.status == search.STATUS_EXHAUSTED
    check = verify.verify_record(world, _text(result.trace), kind="prefix")
    assert check.first_error is None
    assert check.search_outcome == verify.OUTCOME_FRONTIER_EMPTY
    assert check.frontier_remaining == 0


def test_truncated_generation_is_a_legal_prefix(tree, body):
    """生成被截断：末行不完整时标 pending_event，已出现部分仍然检查。"""
    text = "\n".join(body[:4]) + "\nNEW n3 VIA r1 STAT"
    result = verify.verify_record(tree, text, kind="prefix")
    assert result.parse_status == verify.PARSE_PENDING
    assert result.first_error is None
    assert result.search_outcome in (
        verify.OUTCOME_STOPPED_EARLY,
        verify.OUTCOME_BUDGET_EXHAUSTED,
    )


def test_within_budget_separate_from_validity(tree):
    """搜索完整正确但超出预算：三个字段分别记录。"""
    full = search.solve(tree, "forward-bfs")
    assert full.cost.checked_nodes > 4
    result = verify.verify_record(tree, _text(full.trace), kind="search", budget=4)
    assert result.plan_valid is True
    assert result.search_valid is True
    assert result.within_budget is False
    assert result.search_outcome == verify.OUTCOME_SOLVED


def test_solved_outcome_requires_end(tree, body):
    """有 GOAL 但没有 END：不算 solved。"""
    goal_only = body + ["PLAN r0 r2 r6"]
    result = verify.verify_record(tree, _text(goal_only), kind="search")
    assert result.search_outcome != verify.OUTCOME_SOLVED
    assert result.search_valid is False


def test_example_world_all_modes_round_trip():
    """四种模式的正确输出都应判为 solved。"""
    world = example_world()
    for mode in search.MODES:
        result = search.solve(world, mode)
        check = verify.verify_record(
            world, _text(result.trace), kind="search", modes=search.MODES
        )
        assert check.search_outcome == verify.OUTCOME_SOLVED
        assert check.plan_valid and check.search_valid
        assert mode in check.compatible_modes

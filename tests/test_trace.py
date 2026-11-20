"""轨迹协议与模式判别（第 12.3 节 4–8、18–21）。"""

from __future__ import annotations

import pytest

from symplan import env, search, verify
from tests.fixtures import (
    BACKWARD_TRACE,
    FORWARD_TRACE,
    binary_tree_world,
    cycle_world,
    deadend_world,
    example_world,
)


def test_doc_traces_reproduced_exactly():
    """第 4.2 节两条轨迹逐行等于求解器输出。"""
    world = example_world()
    assert search.solve(world, "forward-bfs").trace == FORWARD_TRACE
    assert search.solve(world, "forward-dfs").trace == FORWARD_TRACE
    assert search.solve(world, "backward-bfs").trace == BACKWARD_TRACE


def test_backward_plan_passes_forward_verifier():
    world = example_world()
    result = search.solve(world, "backward-bfs")
    assert result.plan == ["r0", "r1"]
    assert env.verify_plan(world, result.plan).valid


def test_direction_keyword_per_mode():
    world = cycle_world()
    for mode in search.MODES:
        result = search.solve(world, mode)
        keyword = "STATE" if mode.startswith("forward") else "NEED"
        other = "NEED" if keyword == "STATE" else "STATE"
        assert any(f" {keyword} " in line for line in result.trace)
        assert all(f" {other} " not in line for line in result.trace)


def test_keyword_swap_is_detected():
    world = example_world()
    swapped = [line.replace("STATE", "NEED") for line in FORWARD_TRACE]
    result = verify.verify_search(world, swapped)
    assert result.direction == "backward"
    assert result.compatible_modes == []
    assert result.first_error.code == env.ROOT_MISMATCH


def test_mixed_keywords_is_direction_mismatch():
    world = example_world()
    mixed = list(FORWARD_TRACE)
    mixed[1] = mixed[1].replace("STATE", "NEED")
    result = verify.verify_search(world, mixed)
    assert result.parse_status == verify.PARSE_ERROR_STATUS
    assert result.first_error.code == env.DIRECTION_MISMATCH


def test_direction_narrows_compatible_set_to_two():
    world = example_world()
    result = verify.verify_search(world, FORWARD_TRACE, modes=search.MODES)
    assert result.compatible_modes == ["forward-bfs", "forward-dfs"]
    assert result.mode_errors["backward-bfs"] == env.DIRECTION_MISMATCH
    assert result.mode_errors["backward-dfs"] == env.DIRECTION_MISMATCH


def test_every_mode_trace_verifies_against_its_own_mode():
    for world in (example_world(), cycle_world(), binary_tree_world(3), deadend_world(6)):
        for mode in search.MODES:
            result = search.solve(world, mode)
            if not result.solved:
                continue
            check = verify.verify_search(world, result.trace, modes=search.MODES)
            assert check.search_valid, (mode, check.first_error)
            assert check.plan_valid
            assert check.plan_trace_consistent
            assert mode in check.compatible_modes


def test_bfs_and_dfs_distinguished_on_tree():
    world = binary_tree_world(4)
    pair = ("forward-bfs", "forward-dfs")
    for mode in pair:
        result = search.solve(world, mode)
        check = verify.verify_search(world, result.trace, modes=pair)
        assert check.compatible_modes == [mode]
        assert check.first_unique_line is not None


def test_shared_prefix_compatible_with_both_then_diverges():
    world = binary_tree_world(4)
    bfs = search.solve(world, "forward-bfs").trace
    dfs = search.solve(world, "forward-dfs").trace
    common = 0
    for a, b in zip(bfs, dfs):
        if a != b:
            break
        common += 1
    assert common >= 3
    pair = ("forward-bfs", "forward-dfs")
    shared = verify.verify_search(world, bfs[:common], modes=pair, prefix=True)
    assert set(shared.compatible_modes) == set(pair)
    diverged = verify.verify_search(world, bfs[: common + 1], modes=pair, prefix=True)
    assert diverged.compatible_modes == ["forward-bfs"]
    assert diverged.first_divergence_line == common + 1


def test_legal_nonfixed_schedule_separated_from_process_error():
    """合法但非 FIFO/LIFO 的 frontier 顺序：通用检查通过，固定模式不相容。"""
    world = binary_tree_world(3)
    bfs = search.solve(world, "forward-bfs")
    # 把第二次 EXPAND 换成 frontier 中的另一个节点，并重放其后继。
    reordered = _replay_with_order(world, [0, 2])
    check = verify.verify_search(world, reordered, modes=search.MODES, prefix=True)
    assert check.first_error is None
    assert check.compatible_modes == []
    assert check.mode_errors["forward-bfs"] == env.BAD_FRONTIER_CHOICE
    assert check.mode_errors["forward-dfs"] == env.BAD_FRONTIER_CHOICE
    assert bfs.trace[0] == reordered[0]


def _replay_with_order(world, order):
    """按给定的节点取出顺序生成合法事件前缀，用于构造非固定调度的轨迹。"""
    lines: list[str] = []
    vectors = [world.initial]
    visited = {world.initial: 0}
    for idx in order:
        vec = vectors[idx]
        lines.append(f"EXPAND n{idx} STATE {search.format_vector(vec)}")
        if env.meets_goal(vec, world.goal):
            lines.append(f"GOAL n{idx}")
            break
        for rule in world.rules:
            child = env.step(vec, rule)
            if child is None:
                continue
            known = visited.get(child)
            if known is not None:
                lines.append(f"DUP n{known} VIA {rule.id}")
                continue
            new_idx = len(vectors)
            vectors.append(child)
            visited[child] = new_idx
            lines.append(f"NEW n{new_idx} VIA {rule.id} STATE {search.format_vector(child)}")
        lines.append(f"CLOSE n{idx}")
    return lines


def test_budget_boundary_inclusive():
    """第 B 个节点为目标算成功，第 B+1 个不算。"""
    world = deadend_world(10)
    dfs_cost = search.solve(world, "forward-dfs").cost.checked_nodes
    assert dfs_cost == 12
    exact = search.solve(world, "forward-dfs", search.Limits(max_expansions=12))
    assert exact.status == search.STATUS_SOLVED
    assert exact.cost.checked_nodes == 12
    short = search.solve(world, "forward-dfs", search.Limits(max_expansions=11))
    assert short.status == search.STATUS_BUDGET
    assert short.cost.checked_nodes == 11


def test_budget_exceeded_trace_ends_with_close():
    world = deadend_world(20)
    result = search.solve(world, "forward-dfs", search.Limits(max_expansions=5))
    assert result.status == search.STATUS_BUDGET
    assert result.plan is None
    assert result.trace[-1].startswith("CLOSE ")
    assert sum(1 for line in result.trace if line.startswith("EXPAND ")) == 5
    # 失败轨迹可直接作为 prefix 输入。
    check = verify.verify_search(world, result.trace, prefix=True)
    assert check.first_error is None
    assert check.parse_status == verify.PARSE_PREFIX


def test_state_limit_status():
    world = binary_tree_world(5)
    result = search.solve(world, "forward-bfs", search.Limits(max_expansions=10000, max_states=8))
    assert result.status == search.STATUS_STATE_LIMIT
    assert result.plan is None


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda t: [ln for ln in t if not ln.startswith("NEW n1")], env.MISSING_SUCCESSOR),
        (
            lambda t: [ln.replace("STATE 0 0 1 1 0", "STATE 0 0 1 0 1") for ln in t],
            env.WRONG_STATE,
        ),
        (lambda t: [ln for ln in t if ln != "CLOSE n0"], env.UNEXPECTED_EVENT),
        (lambda t: [ln.replace("GOAL n2", "GOAL n1") for ln in t], env.UNEXPECTED_EVENT),
        (
            lambda t: t[:2] + ["NEW n9 VIA r1 STATE 0 0 0 0 1"] + t[2:],
            env.EXTRA_SUCCESSOR,
        ),
        (
            lambda t: ["DUP n0 VIA r0" if ln.startswith("NEW n1 VIA r0") else ln for ln in t],
            env.DUP_TARGET_MISMATCH,
        ),
        (lambda t: [ln.replace("VIA r0", "VIA r7") for ln in t], env.UNKNOWN_RULE),
        (lambda t: [ln.replace("NEW n1", "NEW n5") for ln in t], env.BAD_NODE_ORDER),
    ],
)
def test_tampered_traces_detected(mutate, code):
    world = example_world()
    result = verify.verify_search(world, mutate(list(FORWARD_TRACE)))
    assert result.search_valid is False
    assert result.first_error.code == code, result.first_error


def test_plan_only_claims_no_mode():
    world = example_world()
    result = verify.verify_record(world, "<think>\n\n</think>\n\nPLAN r0 r1\nEND\n", kind="plan")
    assert result.plan_valid is True
    assert result.search_valid is None
    assert result.plan_trace_consistent is None
    assert result.compatible_modes is None
    assert result.checked_nodes is None
    assert result.within_budget is None
    assert result.direction is None


def test_plan_only_invalid_plan():
    world = example_world()
    result = verify.verify_record(world, "PLAN r1 r0\nEND\n", kind="plan")
    assert result.plan_valid is False
    assert result.first_error.code == env.INSUFFICIENT_RESOURCE


def test_parse_status_values():
    world = example_world()
    text = "\n".join(FORWARD_TRACE) + "\n"
    assert verify.verify_record(world, text, kind="search").parse_status == verify.PARSE_COMPLETE

    partial = "\n".join(FORWARD_TRACE[:3]) + "\n"
    assert verify.verify_record(world, partial, kind="prefix").parse_status == verify.PARSE_PREFIX

    pending = "\n".join(FORWARD_TRACE[:2]) + "\nCLOS"
    res = verify.verify_record(world, pending, kind="prefix")
    assert res.parse_status == verify.PARSE_PENDING
    assert res.first_error is None

    broken = "EXPAND n0 STATE 1 1 1 0\n"
    res = verify.verify_record(world, broken, kind="search")
    assert res.parse_status == verify.PARSE_ERROR_STATUS
    assert res.first_error.code == env.VECTOR_LENGTH


def test_search_kind_rejects_incomplete():
    world = example_world()
    partial = "\n".join(FORWARD_TRACE[:3]) + "\n"
    result = verify.verify_record(world, partial, kind="search")
    assert result.search_valid is False
    assert result.first_error.code == env.INCOMPLETE_TRACE


def test_prefix_leaves_completeness_fields_null():
    world = example_world()
    partial = "\n".join(FORWARD_TRACE[:4]) + "\n"
    result = verify.verify_record(world, partial, kind="prefix")
    assert result.plan_valid is None
    assert result.search_valid is None
    assert result.plan_trace_consistent is None
    assert result.compatible_modes == ["forward-bfs", "forward-dfs"]
    assert result.checked_nodes == 2


def test_prefix_still_reports_determined_error():
    world = example_world()
    bad = ["EXPAND n0 STATE 1 1 1 0 0", "NEW n1 VIA r0 STATE 9 9 9 9 9"]
    result = verify.verify_record(world, "\n".join(bad), kind="prefix")
    assert result.first_error.code == env.WRONG_STATE


def test_valid_plan_with_invalid_search():
    """搜索过程错误但最终方案有效：plan_valid=true、search_valid=false。"""
    world = example_world()
    trace = list(FORWARD_TRACE)
    trace.insert(1, "NEW n9 VIA r1 STATE 0 0 0 0 1")
    result = verify.verify_search(world, trace)
    assert result.search_valid is False
    plan_only = verify.verify_record(world, "PLAN r0 r1\nEND\n", kind="plan")
    assert plan_only.plan_valid is True


def test_within_budget_independent_of_validity():
    world = example_world()
    tight = verify.verify_search(world, FORWARD_TRACE, budget=2)
    assert tight.plan_valid is True
    assert tight.search_valid is True
    assert tight.within_budget is False
    loose = verify.verify_search(world, FORWARD_TRACE, budget=3)
    assert loose.within_budget is True


def test_prefill_restoration_is_idempotent():
    world = example_world()
    suffix = "PLAN r0 r1\nEND\n"
    direct = verify.verify_record(world, suffix, kind="plan", prefill="nonthink")
    assert direct.plan_valid
    already = "<think>\n\n</think>\n\n" + suffix
    assert verify.restore_prefill(already, "nonthink") == already


def test_plan_inside_think_block_rejected():
    world = example_world()
    text = "<think>\nPLAN r0 r1\nEND\n</think>\n\n"
    result = verify.verify_record(world, text, kind="search")
    assert result.parse_status == verify.PARSE_ERROR_STATUS
    assert result.first_error.code == env.UNEXPECTED_EVENT


def test_invalid_final_plan_does_not_invalidate_correct_search_events():
    world = example_world()
    trace = [line.replace("PLAN r0 r1", "PLAN r1 r0") for line in FORWARD_TRACE]
    result = verify.verify_search(world, trace)
    assert result.search_valid is True and result.plan_valid is False
    assert result.plan_trace_consistent is False and not result.passed
    assert result.first_error.code == env.INSUFFICIENT_RESOURCE

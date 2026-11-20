"""向量的两种编码、轨迹行数上限、hack 标记（第 4.4、8.2、8.4 节，附录 B）。"""

from __future__ import annotations

import pytest

from symplan import env, qa, search, verify
from symplan.env import Rule, World
from tests.fixtures import binary_tree_world, example_world


# --- 向量编码 --------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", env.VECTOR_FORMATS)
def test_vector_roundtrip(fmt):
    symbols = ("A", "B", "C", "G")
    for vec in ((1, 0, 2, 0), (0, 0, 0, 0), (3, 1, 1, 1)):
        text = env.format_vector(vec, symbols, fmt)
        parsed, got_fmt, code = env.parse_vector(text.split(" "), symbols)
        assert code is None
        assert parsed == vec
        assert got_fmt == fmt


def test_sparse_rendering_is_canonical():
    symbols = ("A", "B", "C", "G")
    assert env.format_vector((1, 0, 2, 0), symbols, "sparse") == "A:1 C:2"
    assert env.format_vector((0, 0, 0, 0), symbols, "sparse") == "-"
    assert env.format_vector((1, 0, 2, 0), symbols, "dense") == "1 0 2 0"


@pytest.mark.parametrize(
    "tokens, code",
    [
        (["A:1", "A:2"], env.VECTOR_SYMBOL_DUPLICATE),
        (["C:2", "A:1"], env.VECTOR_SYMBOL_ORDER),
        (["Z:1"], env.VECTOR_SYMBOL_UNKNOWN),
        (["A:0"], env.PARSE_ERROR),
        (["A:x"], env.PARSE_ERROR),
        (["1", "0", "2"], env.VECTOR_LENGTH),
        (["1", "0", "2", "x"], env.PARSE_ERROR),
    ],
)
def test_malformed_vectors_have_distinct_codes(tokens, code):
    symbols = ("A", "B", "C", "G")
    parsed, _, got = env.parse_vector(tokens, symbols)
    assert parsed is None
    assert got == code


def test_format_detection_is_unambiguous():
    assert env.detect_vector_format(["1", "0"]) == env.VECTOR_DENSE
    assert env.detect_vector_format(["A:1"]) == env.VECTOR_SPARSE
    assert env.detect_vector_format(["-"]) == env.VECTOR_SPARSE


@pytest.mark.parametrize("fmt", env.VECTOR_FORMATS)
def test_solver_output_verifies_under_both_formats(fmt):
    for world in (example_world(), binary_tree_world(3)):
        for mode in search.MODES:
            result = search.solve(world, mode, search.Limits(vector_format=fmt))
            if not result.solved:
                continue
            check = verify.verify_record(
                world, "\n".join(result.trace) + "\n", kind="search", modes=search.MODES
            )
            assert check.vector_format == fmt
            assert check.search_outcome == verify.OUTCOME_SOLVED
            assert check.plan_valid and check.search_valid
            assert mode in check.compatible_modes


def test_both_formats_give_the_same_plan():
    world = binary_tree_world(3)
    dense = search.solve(world, "forward-bfs", search.Limits(vector_format="dense"))
    sparse = search.solve(world, "forward-bfs", search.Limits(vector_format="sparse"))
    assert dense.plan == sparse.plan
    assert dense.cost.to_dict() == sparse.cost.to_dict()
    assert len(dense.trace) == len(sparse.trace)


def test_mixing_formats_in_one_trace_is_rejected():
    world = example_world()
    dense = search.solve(world, "forward-bfs", search.Limits(vector_format="dense"))
    sparse = search.solve(world, "forward-bfs", search.Limits(vector_format="sparse"))
    mixed = [dense.trace[0]] + sparse.trace[1:]
    result = verify.verify_record(world, "\n".join(mixed) + "\n", kind="search")
    assert result.parse_status == verify.PARSE_ERROR_STATUS
    assert result.first_error.code == env.VECTOR_FORMAT_MIXED


def test_prompt_variant_records_the_format():
    world = example_world()
    for fmt, suffix in (("dense", "search-forward"), ("sparse", "search-forward-sparse")):
        result = search.solve(world, "forward-bfs", search.Limits(vector_format=fmt))
        record = qa.render_qa(world, result, view="search", thinking="on")
        assert record.prompt_variant == suffix
        marker = "<counts> gives one integer" if fmt == "dense" else "whose count is not zero"
        assert marker in record.question


def test_sparse_question_describes_the_empty_state():
    world = example_world()
    question = qa.render_question(world, view="search", vector_format="sparse")
    assert "written as a single -" in question


def test_sparse_is_shorter_in_characters():
    world = binary_tree_world(4)
    dense = search.solve(world, "forward-bfs", search.Limits(vector_format="dense"))
    sparse = search.solve(world, "forward-bfs", search.Limits(vector_format="sparse"))
    assert sum(len(x) for x in sparse.trace) < sum(len(x) for x in dense.trace)


# --- 轨迹行数上限 ----------------------------------------------------------- #


def test_trace_line_cap_stops_the_search():
    world = binary_tree_world(5)
    result = search.solve(world, "forward-bfs", search.Limits(max_trace_lines=12))
    assert result.status == search.STATUS_TRACE_LIMIT
    assert result.plan is None
    assert len(result.trace) <= 12
    # 截断处的事件前缀仍然合法
    check = verify.verify_record(world, "\n".join(result.trace) + "\n", kind="prefix")
    assert check.first_error is None


def test_trace_line_cap_zero_means_unlimited():
    world = binary_tree_world(3)
    capped = search.solve(world, "forward-bfs", search.Limits(max_trace_lines=0))
    plain = search.solve(world, "forward-bfs")
    assert capped.trace == plain.trace
    assert capped.status == search.STATUS_SOLVED


def test_trace_limit_is_reported_in_limits():
    world = binary_tree_world(4)
    result = search.solve(
        world, "forward-bfs", search.Limits(max_trace_lines=8, vector_format="sparse")
    )
    payload = result.to_dict()
    assert payload["limits"]["max_trace_lines"] == 8
    assert payload["limits"]["vector_format"] == "sparse"
    assert search.SolveResult.from_dict(payload).limits.max_trace_lines == 8


# --- hack 标记 -------------------------------------------------------------- #


def _two_route_world() -> World:
    """r0;r1 与 r2;r3 都是有效方案。"""
    world = World(
        symbols=("A", "B", "C", "G"),
        initial=(2, 0, 0, 0),
        goal=(0, 0, 0, 1),
        rules=(
            Rule("r0", (1, 0, 0, 0), (0, 1, 0, 0)),
            Rule("r1", (0, 1, 0, 0), (0, 0, 0, 1)),
            Rule("r2", (1, 0, 0, 0), (0, 0, 1, 0)),
            Rule("r3", (0, 0, 1, 0), (0, 0, 0, 1)),
        ),
        world_id="w_two",
        group_id="g_two",
    )
    env.validate_world(world)
    return world


def test_clean_output_is_not_hack():
    world = binary_tree_world(3)
    result = search.solve(world, "forward-bfs")
    check = verify.verify_record(world, "\n".join(result.trace) + "\n", kind="search")
    assert check.hack is False
    assert check.hack_kind is None


def test_process_error_with_valid_plan_is_hack():
    world = binary_tree_world(3)
    trace = search.solve(world, "forward-bfs").trace
    body = [ln for ln in trace if not (ln.startswith("PLAN") or ln == "END")]
    broken = [ln for ln in body if not ln.startswith("NEW n2 ")]
    text = "\n".join(broken + ["PLAN r0 r2 r6", "END"]) + "\n"
    check = verify.verify_record(world, text, kind="search")
    assert check.hack is True
    assert check.hack_kind == verify.HACK_PROCESS_ERROR
    assert check.plan_valid is True
    assert check.search_valid is False


def test_truncated_search_with_valid_plan_is_hack():
    world = binary_tree_world(3)
    trace = search.solve(world, "forward-bfs").trace
    body = [ln for ln in trace if not (ln.startswith("PLAN") or ln == "END")]
    text = "\n".join(body[:3] + ["PLAN r0 r2 r6", "END"]) + "\n"
    check = verify.verify_record(world, text, kind="search")
    assert check.hack is True
    assert check.plan_valid is True


def test_different_valid_plan_is_allowed_and_consistency_is_diagnostic():
    world = _two_route_world()
    result = search.solve(world, "forward-bfs")
    body = [ln for ln in result.trace if not (ln.startswith("PLAN") or ln == "END")]
    alt = ["r2", "r3"]
    assert alt != result.plan
    assert env.verify_plan(world, alt).valid
    text = "\n".join(body + ["PLAN " + " ".join(alt), "END"]) + "\n"
    check = verify.verify_record(world, text, kind="search")
    assert check.hack is False
    assert check.hack_kind is None
    assert check.plan_trace_consistent is False
    assert check.search_valid is True and check.plan_valid is True and check.passed
    assert check.plan_shortest is None


def test_wrong_process_and_wrong_plan_is_not_hack():
    world = binary_tree_world(3)
    trace = search.solve(world, "forward-bfs").trace
    body = [ln for ln in trace if not (ln.startswith("PLAN") or ln == "END")]
    broken = [ln for ln in body if not ln.startswith("NEW n2 ")]
    text = "\n".join(broken + ["PLAN r0 r0", "END"]) + "\n"
    check = verify.verify_record(world, text, kind="search")
    assert check.hack is False
    assert check.plan_valid is False


def test_hack_is_null_for_plan_only_input():
    world = binary_tree_world(3)
    check = verify.verify_record(world, "PLAN r0 r2 r6\nEND\n", kind="plan")
    assert check.hack is None
    assert check.hack_kind is None


def _longer_route_world():
    return World(symbols=("A", "B", "G"), initial=(1, 0, 0), goal=(0, 0, 1),
                 rules=(Rule("r0", (1, 0, 0), (0, 0, 1)),
                        Rule("r1", (1, 0, 0), (0, 1, 0)),
                        Rule("r2", (0, 1, 0), (0, 0, 1)),
                        Rule("r3", (0, 0, 1), (0, 1, 0))))


def test_longer_valid_final_plan_is_successful_without_optimality_search(monkeypatch):
    world = _longer_route_world()
    trace = search.solve(world, "forward-bfs").trace
    text = "\n".join([*trace[:-2], "PLAN r1 r2", "END"])
    monkeypatch.setattr(env, "shortest_distance", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected shortest-path search")))
    check = verify.verify_record(world, text, kind="search")
    assert check.plan_valid and check.search_valid and check.passed
    assert check.plan_trace_consistent is False and check.hack is False
    assert check.first_error is None and "forward-bfs" in check.compatible_modes
    assert check.plan_steps == 2 and check.plan_shortest is None
    assert check.plan_waste is None



def test_redundant_valid_steps_are_allowed_without_shortest_search(monkeypatch):
    monkeypatch.setattr(env, "shortest_distance", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unexpected shortest-path search")))
    check = verify.verify_record(_longer_route_world(), "PLAN r0 r3 r2\nEND", kind="plan")
    assert check.plan_valid and check.passed
    assert check.first_goal_step == 1 and check.plan_steps == 3
    assert check.plan_shortest is None and check.plan_waste is None

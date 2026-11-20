"""Q 模板、think/非 think 渲染与目标权重（第 12.3 节 11、12、15、16、24）。"""

from __future__ import annotations

import pytest

from symplan import qa, search, verify
from tests.fixtures import example_world


def test_structured_prompt_contains_only_world_fields_for_all_modes():
    world = example_world()
    questions = set()
    for mode in search.MODES:
        result = search.solve(world, mode)
        record = qa.render_qa(world, result, view="search", thinking="on",
                              directions=(search.direction_of(mode),), prompt_style="structured")
        questions.add(record.question)
        assert record.question == qa.render_header(world)
        assert record.prompt_variant == "search-structured-dense-v2"
        assert record.assistant == qa.render_assistant(result, "search", "on")
        assert verify.verify_record(world, record.assistant, kind="search").passed
    assert len(questions) == 1


def test_structured_prompt_explicit_controls_stay_in_header_only():
    world = example_world()
    q = qa.render_question(world, view="search", budget=64, teacher_mode="forward-bfs",
                           prompt_style="structured")
    assert q == qa.render_header(world, budget=64, teacher_mode="forward-bfs")
    assert qa.TASK_TEXT not in q and qa.SCHEDULE_FIFO_LINE not in q
    with pytest.raises(ValueError):
        qa.render_question(world, view="search", prompt_style="unknown")


def test_prompt_revision_changes_hash_without_changing_answer():
    world = example_world()
    result = search.solve(world, "forward-bfs")
    old = qa.render_qa(world, result, view="search", thinking="on", prompt_style="detailed")
    new = qa.render_qa(world, result, view="search", thinking="on", prompt_style="structured")
    assert old.assistant == new.assistant and old.record_id == new.record_id
    assert old.target_hash != new.target_hash
    assert len(new.question) < len(old.question)


def test_question_contains_header_and_output_spec():
    world = example_world()
    question = qa.render_question(world, view="plan")
    assert question.startswith("RESOURCES A B C D G\n")
    assert "INITIAL A:1 B:1 C:1" in question
    assert "GOAL G:1" in question
    assert "RULES\nr0: A:1 B:1 -> D:1\nr1: C:1 D:1 -> G:1" in question
    assert question.rstrip().endswith("END")
    assert "SEARCH_BUDGET" not in question
    assert "SEARCH_MODE" not in question


def test_no_leakage_of_witness_or_class_or_mode():
    world = example_world()
    world.meta["qualification"] = {"class": "first-only", "budget": 64}
    for view, thinking in ((("plan"), "off"), ("search", "on")):
        question = qa.render_question(world, view=view, directions=("forward",))
        for banned in ("witness", "r0 r1", "first-only", "forward-bfs", "forward-dfs", "64"):
            assert banned not in question, (view, banned)


def test_teacher_prompt_mode_adds_mode_and_schedule():
    world = example_world()
    fifo = qa.render_question(world, view="search", teacher_mode="forward-bfs")
    lifo = qa.render_question(world, view="search", teacher_mode="forward-dfs")
    assert "SEARCH_MODE forward-bfs" in fifo
    assert qa.SCHEDULE_FIFO_LINE in fifo
    assert qa.SCHEDULE_LIFO_LINE in lifo


def test_budget_shown_only_when_given():
    world = example_world()
    with_budget = qa.render_question(world, view="search", budget=64)
    assert "SEARCH_BUDGET 64" in with_budget
    assert qa.BUDGET_LINE in with_budget


def test_event_syntax_section_follows_mode_directions():
    world = example_world()
    fwd = qa.render_question(world, view="search", directions=("forward",))
    bwd = qa.render_question(world, view="search", directions=("backward",))
    both = qa.render_question(world, view="search", directions=("forward", "backward"))
    assert "STATE <counts>" in fwd and "NEED <counts>" not in fwd
    assert "NEED <counts>" in bwd and "STATE <counts>" not in bwd
    assert "STATE <counts>" in both and "NEED <counts>" in both
    assert both.index("STATE <counts>") < both.index("NEED <counts>")


def test_prompt_variant_changes_target_hash():
    world = example_world()
    result = search.solve(world, "forward-bfs")
    variants = {}
    for directions in (("forward",), ("forward", "backward")):
        record = qa.render_qa(
            world, result, view="search", thinking="on", directions=directions
        )
        variants[record.prompt_variant] = record.target_hash
    assert set(variants) == {"search-forward", "search-both"}
    assert len(set(variants.values())) == 2
    # 同一变体下重复导出稳定。
    again = qa.render_qa(world, result, view="search", thinking="on", directions=("forward",))
    assert again.target_hash == variants["search-forward"]


def test_illegal_protocol_combinations_rejected():
    for view, thinking in (("plan", "on"), ("search", "off")):
        with pytest.raises(ValueError):
            qa.check_protocol(view, thinking)
    qa.check_protocol("plan", "off")
    qa.check_protocol("search", "on")


def test_nonthink_and_think_share_the_same_plan():
    world = example_world()
    result = search.solve(world, "forward-bfs")
    nonthink = qa.render_assistant(result, "plan", "off")
    think = qa.render_assistant(result, "search", "on")
    assert nonthink == "<think>\n\n</think>\n\nPLAN r0 r1\nEND"
    assert think.startswith("<think>\nEXPAND n0 STATE 1 1 1 0 0\n")
    assert think.endswith("</think>\n\nPLAN r0 r1\nEND")
    assert nonthink.split("</think>")[1] == think.split("</think>")[1]
    assert think.count("<think>") == 1 and think.count("</think>") == 1
    assert nonthink.count("<think>") == 1 and nonthink.count("</think>") == 1
    # PLAN/END 不重复出现在 think 块内。
    assert "PLAN" not in think.split("</think>")[0]


def test_failed_result_cannot_be_rendered():
    world = example_world()
    result = search.solve(world, "forward-bfs", search.Limits(max_expansions=1))
    assert not result.solved
    with pytest.raises(ValueError):
        qa.render_assistant(result, "plan", "off")


def test_train_and_inference_prefix_align():
    world = example_world()
    result = search.solve(world, "forward-bfs")
    for view, thinking in (("plan", "off"), ("search", "on")):
        record = qa.render_qa(world, result, view=view, thinking=thinking)
        train = qa.render_train_text(record.question, record.assistant)
        inference = qa.render_inference_text(record.question, view)
        assert train.startswith(inference)
        suffix = train[len(inference) :]
        assert inference + suffix == train
        assert train.endswith("<|im_end|>")
        assert train.count("<think>") == 1
        assert train.count("</think>") == 1
        assert record.assistant.startswith(record.assistant_prefill)


def test_rendered_qa_roundtrips_through_verify():
    world = example_world()
    for mode in search.MODES:
        result = search.solve(world, mode)
        directions = (search.direction_of(mode),)
        record = qa.render_qa(
            world, result, view="search", thinking="on", directions=directions
        )
        check = verify.verify_record(
            world, record.assistant, kind="search", modes=search.MODES
        )
        assert check.plan_valid and check.search_valid and check.plan_trace_consistent
        assert mode in check.compatible_modes


def test_target_hash_merges_identical_assistant_text():
    world = example_world()
    bfs = search.solve(world, "forward-bfs")
    dfs = search.solve(world, "forward-dfs")
    assert bfs.trace == dfs.trace
    a = qa.render_qa(world, bfs, view="search", thinking="on")
    b = qa.render_qa(world, dfs, view="search", thinking="on")
    assert a.source_mode != b.source_mode
    assert a.target_hash == b.target_hash


def test_target_hash_distinguishes_different_traces():
    from tests.fixtures import binary_tree_world

    world = binary_tree_world(3)
    a = qa.render_qa(world, search.solve(world, "forward-bfs"), view="search", thinking="on")
    b = qa.render_qa(world, search.solve(world, "forward-dfs"), view="search", thinking="on")
    assert a.target_hash != b.target_hash


def test_target_hash_canonicalisation_ignores_trailing_whitespace():
    assert qa.target_hash("w", "a\nb", "c") == qa.target_hash("w", "a  \nb\n\n", "\nc  ")


def test_teacher_prompt_narrows_event_section_to_its_direction():
    world = example_world()
    for mode in search.MODES:
        result = search.solve(world, mode)
        record = qa.render_qa(
            world,
            result,
            view="search",
            thinking="on",
            directions=("forward", "backward"),
            prompt_mode=qa.PROMPT_MODE_TEACHER,
        )
        direction = search.direction_of(mode)
        assert record.prompt_variant == f"search-{direction}"
        question = record.question
        assert f"SEARCH_MODE {mode}" in question
        keyword = "STATE" if direction == "forward" else "NEED"
        other = "NEED" if direction == "forward" else "STATE"
        assert f"{keyword} <counts>" in question
        assert f"{other} <counts>" not in question

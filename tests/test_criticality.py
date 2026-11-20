"""步骤分类、修复代价与偏离后果（第 6.9 节，第 12.3 节 30–31、33）。"""

from __future__ import annotations

import pytest

from symplan import env, generate, search, verify
from symplan.env import World
from tests.fixtures import binary_tree_world, cycle_world, example_world


def test_shortest_distance_counts_steps():
    world = example_world()
    assert env.shortest_distance(world, world.initial) == (env.REACHABLE, 2)
    assert env.shortest_distance(world, (0, 0, 1, 1, 0)) == (env.REACHABLE, 1)
    assert env.shortest_distance(world, (0, 0, 0, 0, 1)) == (env.REACHABLE, 0)
    # 资源被消耗后目标不可达
    assert env.shortest_distance(world, (1, 1, 0, 0, 0)) == (env.UNREACHABLE, None)


def test_shortest_distance_undetermined_on_small_cap():
    world = binary_tree_world(5)
    status, distance = env.shortest_distance(world, world.initial, max_states=4)
    assert status == env.UNDETERMINED
    assert distance is None
    assert env.goal_reachable(world, world.initial, max_states=4) is None


def test_goal_reachable_matches_distance():
    for world in (example_world(), cycle_world(), binary_tree_world(3)):
        status, _ = env.shortest_distance(world, world.initial)
        assert env.goal_reachable(world, world.initial) == (status == env.REACHABLE)


def test_cycle_world_step_kinds_and_waste():
    """循环世界的最后一步：绕一圈回来要多走 3 步，属于浪费步。"""
    world = cycle_world()
    outcome = env.plan_outcome(world, ["r0", "r1", "r3"])
    assert [s.kind for s in outcome.steps] == [
        env.KIND_FORCED,
        env.KIND_FORCED,
        env.KIND_WASTEFUL,
    ]
    assert [s.best_distance for s in outcome.steps] == [3, 2, 1]
    last = outcome.steps[-1]
    assert last.alternatives == 1
    assert last.max_waste == 3
    assert last.fatal == 0
    assert outcome.wasteful_steps == 1
    assert outcome.critical_steps == 0


def test_forced_steps_have_no_alternatives():
    world = example_world()
    outcome = env.plan_outcome(world, ["r0", "r1"])
    assert all(s.kind == env.KIND_FORCED for s in outcome.steps)
    assert all(s.alternatives == 0 for s in outcome.steps)
    assert outcome.forced_steps == 2


def _generated_worlds(n: int = 12):
    cfg = generate.GenConfig(
        family="general",
        structure="mixed",
        level="easy",
        resources=(8, 12),
        initial_units=(3, 5),
        rules=(10, 18),
        steps=(2, 4),
        arity=2,
        distractors=0.25,
        count=n,
        seed=515,
    )
    out = []
    slot = 0
    while len(out) < n and slot < n * 8:
        structure = generate.MAIN_STRUCTURES[slot % len(generate.MAIN_STRUCTURES)]
        candidate = generate.make_candidate(cfg, structure, slot, 0)
        slot += 1
        if isinstance(candidate, generate.Candidate):
            out.append(candidate.world)
    return out


def test_classification_is_a_partition_and_consistent():
    """四类加未判定构成划分；各类与可达性判定一致。"""
    checked = 0
    for world in _generated_worlds():
        plan = world.meta["witness_plan"]
        outcome = env.plan_outcome(world, plan)
        assert len(outcome.steps) == len(plan)
        total = (
            outcome.forced_steps
            + outcome.free_steps
            + outcome.wasteful_steps
            + outcome.critical_steps
            + outcome.unknown_steps
        )
        assert total == len(plan)
        for step in outcome.steps:
            if step.kind == env.KIND_FREE:
                assert step.fatal == 0
                assert all(d.waste == 0 for d in step.deviations)
            elif step.kind == env.KIND_WASTEFUL:
                assert step.fatal == 0
                assert step.wasteful >= 1
            elif step.kind == env.KIND_CRITICAL:
                assert step.fatal >= 1
            elif step.kind == env.KIND_FORCED:
                assert step.applicable == 1
            for dev in step.deviations:
                if dev.waste is not None:
                    assert dev.waste >= 0
                    assert dev.status == env.REACHABLE
                else:
                    assert dev.status in (env.UNREACHABLE, env.UNDETERMINED)
        checked += 1
    assert checked >= 5


def test_fatal_deviation_makes_any_continuation_invalid():
    """必错偏离之后不存在任何有效续写。"""
    found = 0
    for world in _generated_worlds():
        plan = world.meta["witness_plan"]
        outcome = env.plan_outcome(world, plan)
        state = world.initial
        for t, step in enumerate(outcome.steps):
            for dev in step.deviations:
                if not dev.fatal:
                    continue
                rule = world.rule(dev.rule_id)
                assert rule is not None
                nxt = env.step(state, rule)
                assert nxt is not None
                probe = World(
                    symbols=world.symbols,
                    initial=nxt,
                    goal=world.goal,
                    rules=world.rules,
                    world_id="probe",
                )
                # 穷举该状态的全部可达空间仍然找不到方案
                assert not search.solve(probe, "forward-bfs").solved
                assert not env.verify_plan(world, plan[:t] + [rule.id] + plan[t + 1 :]).valid
                found += 1
            taken = world.rule(plan[t])
            assert taken is not None
            nxt = env.step(state, taken)
            assert nxt is not None
            state = nxt
    assert found >= 1


def test_repairable_deviation_has_a_valid_continuation():
    """可修复偏离之后存在有效续写，且其长度等于最短修复长度。"""
    found = 0
    for world in _generated_worlds():
        plan = world.meta["witness_plan"]
        outcome = env.plan_outcome(world, plan)
        state = world.initial
        for t, step in enumerate(outcome.steps):
            for dev in step.deviations:
                if dev.waste is None or dev.waste <= 0:
                    continue
                rule = world.rule(dev.rule_id)
                assert rule is not None
                nxt = env.step(state, rule)
                assert nxt is not None
                probe = World(
                    symbols=world.symbols,
                    initial=nxt,
                    goal=world.goal,
                    rules=world.rules,
                    world_id="probe",
                )
                repair = search.solve(probe, "forward-bfs")
                assert repair.solved
                full = plan[:t] + [rule.id] + (repair.plan or [])
                assert env.verify_plan(world, full).valid
                assert step.best_distance is not None
                assert len(repair.plan or []) - (step.best_distance - 1) == dev.waste
                found += 1
            taken = world.rule(plan[t])
            assert taken is not None
            nxt = env.step(state, taken)
            assert nxt is not None
            state = nxt
    assert found >= 1


def test_suffix_valid_matches_independent_execution():
    for world in _generated_worlds(8):
        plan = world.meta["witness_plan"]
        outcome = env.plan_outcome(world, plan)
        state = world.initial
        for t, step in enumerate(outcome.steps):
            for dev in step.deviations:
                if dev.suffix_valid is None:
                    continue
                candidate = plan[:t] + [dev.rule_id] + plan[t + 1 :]
                assert env.verify_plan(world, candidate).valid == dev.suffix_valid
            taken = world.rule(plan[t])
            assert taken is not None
            nxt = env.step(state, taken)
            assert nxt is not None
            state = nxt


def test_verify_reports_critical_step_and_unrecoverable_step():
    world = cycle_world()
    # 在第 3 步改用 r2 并照抄原后缀：该步为浪费步，后缀失效。
    text = "PLAN r0 r1 r2\nEND\n"
    result = verify.verify_record(world, text, kind="plan", criticality=True)
    assert result.plan_valid is False
    assert result.first_error.code == env.GOAL_NOT_MET
    assert result.first_error_at_critical_step is False
    assert result.first_unrecoverable_step is None  # 目标仍可达，只是没走完

    good = verify.verify_record(world, "PLAN r0 r1 r3\nEND\n", kind="plan", criticality=True)
    assert good.plan_valid is True
    assert good.first_unrecoverable_step is None


def test_verify_reports_unrecoverable_step_on_fatal_prefix():
    world = example_world()
    # r0 之后把 D 再消耗掉是不可能的，这里直接给一个使目标不可达的前缀。
    broken = World(
        symbols=("A", "B", "G"),
        initial=(1, 1, 0),
        goal=(0, 0, 1),
        rules=(
            env.Rule("r0", (1, 0, 0), (0, 1, 0)),
            env.Rule("r1", (1, 1, 0), (0, 0, 1)),
        ),
        world_id="w_fatal",
        group_id="g_fatal",
    )
    env.validate_world(broken)
    assert env.verify_plan(broken, ["r1"]).valid
    result = verify.verify_record(broken, "PLAN r0 r1\nEND\n", kind="plan", criticality=True)
    assert result.plan_valid is False
    assert result.first_unrecoverable_step == 1
    # 首错在第 2 步，该步起点已不可达，属于 doomed 而非关键步；
    # 真正造成损失的是第 1 步，由 first_unrecoverable_step 指出。
    assert result.first_error_at_critical_step is False
    outcome = env.plan_outcome(broken, ["r0"])
    assert outcome.steps[0].kind == env.KIND_CRITICAL


def test_verify_divergence_offset():
    world = binary_tree_world(3)
    bfs = search.solve(world, "forward-bfs")
    record = search.divergence_record(
        world.world_id,
        ("forward-bfs", "forward-dfs"),
        bfs,
        search.solve(world, "forward-dfs"),
    )
    shared = record["shared_lines"]
    tampered = list(bfs.trace)
    # 在分歧位置之后把一行的状态向量改错，构造语义错误而非解析错误
    nxt = next(i for i in range(shared, len(tampered)) if tampered[i].startswith("NEW "))
    head, _, vec = tampered[nxt].partition(" STATE ")
    tampered[nxt] = head + " STATE " + " ".join("9" for _ in vec.split(" "))
    result = verify.verify_search(world, tampered, modes=("forward-bfs", "forward-dfs"))
    assert result.first_error is not None
    assert result.first_error.code == env.WRONG_STATE
    with_offset = verify.verify_record(
        world,
        "\n".join(tampered) + "\n",
        kind="search",
        modes=("forward-bfs", "forward-dfs"),
        shared_lines=shared,
        criticality=True,
    )
    assert with_offset.first_error_offset_to_divergence == (
        with_offset.first_error.line - shared
    )
    # 没有给 shared_lines 时该字段为 null
    plain = verify.verify_search(world, tampered, modes=("forward-bfs", "forward-dfs"))
    assert plain.first_error_offset_to_divergence is None


def test_verify_reports_frontier_size_at_bad_expand():
    world = binary_tree_world(3)
    bfs = search.solve(world, "forward-bfs")
    trace = list(bfs.trace)
    # 把第二次 EXPAND 换成一个不在 frontier 首位的节点
    idx = [i for i, ln in enumerate(trace) if ln.startswith("EXPAND ")][1]
    trace[idx] = trace[idx].replace("EXPAND n1 ", "EXPAND n2 ")
    result = verify.verify_record(
        world,
        "\n".join(trace) + "\n",
        kind="search",
        modes=("forward-bfs",),
        criticality=True,
    )
    assert result.search_valid is False
    assert result.first_error_frontier_size is not None
    assert result.first_error_frontier_size >= 2


def test_criticality_fields_absent_without_flag():
    world = cycle_world()
    result = verify.verify_record(world, "PLAN r0 r1 r2\nEND\n", kind="plan")
    assert result.first_error_at_critical_step is None
    assert result.first_unrecoverable_step is None
    assert result.first_error_frontier_size is None
    assert result.first_error_offset_to_divergence is None


# --- 方案的资源开销记账与验证成本上限 -------------------------------------- #


def _waste_world() -> World:
    """A->B->C->A 构成循环，C->G 到目标；最短方案 3 步。"""
    world = World(
        symbols=("A", "B", "C", "G"),
        initial=(1, 0, 0, 0),
        goal=(0, 0, 0, 1),
        rules=(
            env.Rule("r0", (1, 0, 0, 0), (0, 1, 0, 0)),
            env.Rule("r1", (0, 1, 0, 0), (0, 0, 1, 0)),
            env.Rule("r2", (0, 0, 1, 0), (1, 0, 0, 0)),
            env.Rule("r3", (0, 0, 1, 0), (0, 0, 0, 1)),
        ),
        world_id="w_waste",
        group_id="g_waste",
    )
    env.validate_world(world)
    return world


def test_plan_check_reports_resource_accounting():
    world = _waste_world()
    short = env.verify_plan(world, ["r0", "r1", "r3"])
    assert short.valid
    assert short.steps == 3
    assert short.first_goal_step == 3
    assert short.steps_after_goal == 0
    assert short.state_revisits == 0
    assert short.units_initial == 1 and short.units_final == 1
    assert short.units_consumed == 0
    assert short.units_monotone is True

    loop = env.verify_plan(world, ["r0", "r1", "r2"] * 2 + ["r0", "r1", "r3"])
    assert loop.valid
    assert loop.steps == 9
    assert loop.state_revisits == 6
    assert loop.first_goal_step == 9


def test_plan_check_reports_first_goal_step_before_end():
    """方案在中途达标后继续走，first_goal_step 指出首次达标的位置。"""
    world = World(
        symbols=("A", "B", "C", "G"),
        initial=(1, 0, 0, 0),
        goal=(0, 0, 0, 1),
        rules=(
            env.Rule("r0", (1, 0, 0, 0), (0, 1, 0, 0)),
            env.Rule("r1", (0, 1, 0, 0), (0, 0, 1, 0)),
            env.Rule("r2", (0, 0, 1, 0), (0, 0, 0, 1)),
            env.Rule("r3", (0, 0, 0, 1), (0, 1, 0, 0)),
        ),
        world_id="w_redo",
        group_id="g_redo",
    )
    env.validate_world(world)
    check = env.verify_plan(world, ["r0", "r1", "r2", "r3", "r1", "r2"])
    assert check.valid
    assert check.steps == 6
    assert check.first_goal_step == 3
    assert check.steps_after_goal == 3


def test_verify_reports_plan_waste_against_shortest():
    world = _waste_world()
    text = "PLAN " + " ".join(["r0", "r1", "r2"] * 3 + ["r0", "r1", "r3"]) + "\nEND\n"
    result = verify.verify_record(world, text, kind="plan", criticality=True)
    assert result.plan_valid is True
    assert result.plan_steps == 12
    assert result.plan_shortest == 3
    assert result.plan_waste == 9
    assert result.state_revisits == 9


def test_plan_waste_is_zero_for_shortest_plan():
    world = _waste_world()
    result = verify.verify_record(
        world, "PLAN r0 r1 r3\nEND\n", kind="plan", criticality=True
    )
    assert result.plan_waste == 0
    assert result.state_revisits == 0


def test_plan_too_long_is_rejected_without_execution():
    world = _waste_world()
    text = "PLAN " + " ".join(["r0", "r1", "r2"] * 300) + "\nEND\n"
    result = verify.verify_record(world, text, kind="plan", max_plan_steps=512)
    assert result.plan_valid is False
    assert result.first_error.code == env.PLAN_TOO_LONG
    assert result.plan_steps == 0  # 未执行任何步骤
    assert not result.passed


def test_trace_too_long_is_rejected_before_replay():
    world = _waste_world()
    text = "\n".join(["EXPAND n0 STATE 1 0 0 0"] * 50) + "\n"
    result = verify.verify_record(world, text, kind="search", max_events=10)
    assert result.parse_status == verify.PARSE_ERROR_STATUS
    assert result.first_error.code == env.TRACE_TOO_LONG


def test_caps_can_be_disabled():
    world = _waste_world()
    text = "PLAN " + " ".join(["r0", "r1", "r2"] * 200 + ["r0", "r1", "r3"]) + "\nEND\n"
    result = verify.verify_record(world, text, kind="plan", max_plan_steps=None)
    assert result.plan_valid is True
    assert result.plan_steps == 603


def test_search_view_also_reports_accounting():
    world = _waste_world()
    result = search.solve(world, "forward-bfs")
    assert result.solved
    check = verify.verify_search(world, result.trace)
    assert check.plan_valid is True
    assert check.plan_steps == len(result.plan or [])
    assert check.units_initial == 1
    assert check.state_revisits == 0
    assert check.units_monotone is True

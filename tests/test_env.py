"""环境语义与方案验证（第 12.3 节 1–3、17）。"""

from __future__ import annotations

import pytest

from symplan import env, search
from symplan.env import Rule, World, WorldError
from tests.fixtures import binary_tree_world, cycle_world, deadend_world, example_world


def test_witness_plan_executes():
    world = example_world()
    check = env.verify_plan(world, ["r0", "r1"])
    assert check.valid
    assert check.final_state == (0, 0, 0, 0, 1)


def test_wrong_order_fails_on_resource():
    world = example_world()
    check = env.verify_plan(world, ["r1", "r0"])
    assert not check.valid
    assert check.error.code == env.INSUFFICIENT_RESOURCE
    assert check.error.line == 1


def test_unknown_rule_fails():
    check = env.verify_plan(example_world(), ["r0", "r9"])
    assert not check.valid
    assert check.error.code == env.UNKNOWN_RULE
    assert check.error.line == 2


def test_goal_not_met():
    check = env.verify_plan(example_world(), ["r0"])
    assert not check.valid
    assert check.error.code == env.GOAL_NOT_MET


def test_redundant_steps_allowed():
    """不禁止合法但冗余步骤；长度更长不自动视为错误。"""
    world = cycle_world()
    assert env.verify_plan(world, ["r0", "r1", "r3"]).valid
    assert env.verify_plan(world, ["r0", "r1", "r2", "r0", "r1", "r3"]).valid


def test_empty_plan_only_when_initial_meets_goal():
    world = example_world()
    assert not env.verify_plan(world, []).valid
    trivial = World(
        symbols=("A",), initial=(1,), goal=(1,), rules=(Rule("r0", (1,), (1,)),), world_id="w"
    )
    with pytest.raises(WorldError):
        env.validate_world(trivial)  # 原地不变规则在入口报错


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"initial": (0, 0)}, "初态总资源量"),
        ({"goal": (0, 0)}, "目标总资源量"),
        ({"rules": (Rule("r0", (1, 0), (0, 1)), Rule("r0", (0, 1), (1, 0)))}, "重复"),
        ({"rules": (Rule("r1", (1, 0), (0, 1)), Rule("r0", (0, 1), (1, 0)))}, "升序"),
        ({"rules": (Rule("r0", (1, 0), (1, 0)),)}, "原地不变"),
        ({"rules": (Rule("r0", (0, 0), (0, 1)),)}, "输入为空"),
        ({"rules": (Rule("r0", (1, 0), (0, 0)),)}, "输出为空"),
        ({"rules": (Rule("r0", (1, 0), (0, 2)),)}, "输出总量"),
        ({"initial": (1, -1)}, "负计数"),
        ({"symbols": ("A", "A")}, "重复"),
    ],
)
def test_world_validation_rejects(kwargs, reason):
    base = {
        "symbols": ("A", "B"),
        "initial": (1, 0),
        "goal": (0, 1),
        "rules": (Rule("r0", (1, 0), (0, 1)),),
        "world_id": "w",
    }
    base.update(kwargs)
    with pytest.raises(WorldError) as exc:
        env.validate_world(World(**base))
    assert reason in str(exc.value)


def test_regress_formula_and_total_bound():
    world = example_world()
    rule = world.rule("r1")
    assert env.regress((0, 0, 0, 0, 1), rule, world.total_units) == (0, 0, 1, 1, 0)
    # 回归结果总需求超过 m 时跳过。
    assert env.regress((1, 1, 0, 0, 1), rule, world.total_units) is None


def _enumerate_forward(world: World) -> set[tuple[int, ...]]:
    seen = {world.initial}
    stack = [world.initial]
    while stack:
        state = stack.pop()
        for rule in world.rules:
            nxt = env.step(state, rule)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _random_worlds(n: int = 40):
    import random

    rng = random.Random(4242)
    for _ in range(n):
        n_res = rng.randint(3, 5)
        n_rules = rng.randint(2, 5)
        rules = []
        for i in range(n_rules):
            while True:
                consume = [0] * n_res
                produce = [0] * n_res
                for _ in range(rng.randint(1, 2)):
                    consume[rng.randrange(n_res)] += 1
                for _ in range(rng.randint(1, sum(consume))):
                    produce[rng.randrange(n_res)] += 1
                if sum(produce) <= sum(consume) and tuple(consume) != tuple(produce):
                    break
            rules.append(Rule(f"r{i}", tuple(consume), tuple(produce)))
        initial = [0] * n_res
        for _ in range(rng.randint(1, 4)):
            initial[rng.randrange(n_res)] += 1
        goal = [0] * n_res
        goal[rng.randrange(n_res)] = 1
        world = World(
            symbols=tuple(f"S{i}" for i in range(n_res)),
            initial=tuple(initial),
            goal=tuple(goal),
            rules=tuple(rules),
            world_id="w_rand",
            group_id="g_rand",
        )
        env.validate_world(world)
        yield world


def test_forward_reachability_matches_backward_solvability():
    """对小资源空间穷举，比较正向可达性与反向回归可解性，并执行恢复的方案。"""
    checked = 0
    for world in _random_worlds():
        reachable = _enumerate_forward(world)
        forward_ok = any(env.meets_goal(s, world.goal) for s in reachable)
        for mode in search.MODES:
            result = search.solve(world, mode, search.Limits(max_expansions=20000))
            assert result.status in (search.STATUS_SOLVED, search.STATUS_EXHAUSTED)
            assert result.solved == forward_ok, (world.to_dict(), mode, result.status)
            if result.solved:
                assert env.verify_plan(world, result.plan).valid
        checked += 1
    assert checked > 0


def test_cycle_dedup_and_separate_visited():
    world = cycle_world()
    result = search.solve(world, "forward-bfs")
    assert result.solved
    # A->B->C 三个状态加目标状态，循环不产生新节点。
    assert result.cost.generated_nodes == 4
    assert any(line.startswith("DUP ") for line in result.trace)
    # 两个方向的向量语义不同，关键字也不同。
    backward = search.solve(world, "backward-bfs")
    assert all("NEED" in line for line in backward.trace if "STATE" in line or "NEED" in line)
    assert all("STATE" in line for line in result.trace if "STATE" in line or "NEED" in line)


def test_cost_case_long_deadend():
    world = deadend_world(100)
    assert search.solve(world, "forward-bfs").cost.checked_nodes == 3
    assert search.solve(world, "forward-dfs").cost.checked_nodes == 102


def test_cost_case_binary_tree():
    world = binary_tree_world(5)
    assert search.solve(world, "forward-bfs").cost.checked_nodes == 32
    assert search.solve(world, "forward-dfs").cost.checked_nodes == 6


def test_cost_fields_match_events():
    for world in (example_world(), cycle_world(), binary_tree_world(4)):
        for mode in search.MODES:
            result = search.solve(world, mode)
            n_expand = sum(1 for line in result.trace if line.startswith("EXPAND "))
            n_new = sum(1 for line in result.trace if line.startswith("NEW "))
            assert result.cost.checked_nodes == n_expand
            assert result.cost.generated_nodes == n_new + 1

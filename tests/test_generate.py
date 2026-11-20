"""生成器配额、槽位与结构证据（第 12.3 节 13、17、25）。"""

from __future__ import annotations

import pytest

from symplan import env, generate, search


def _cfg(**kwargs) -> generate.GenConfig:
    base = dict(
        family="general",
        structure="mixed",
        level="easy",
        resources=(8, 12),
        initial_units=(3, 5),
        rules=(10, 18),
        steps=(2, 4),
        arity=2,
        distractors=0.25,
        count=20,
        seed=1,
    )
    base.update(kwargs)
    return generate.GenConfig(**base)


@pytest.mark.parametrize("count", [1, 7, 13, 20, 101, 1000, 15000, 20000])
def test_quota_rounding_is_exact_and_deterministic(count):
    quota = generate.allocate_quota(count)
    assert sum(quota.values()) == count
    assert quota == generate.allocate_quota(count)
    assert list(quota) == list(generate.MAIN_STRUCTURES)


def test_quota_matches_doc_table():
    assert generate.allocate_quota(20000) == {
        "chain": 3000,
        "merge": 5000,
        "alternative": 5000,
        "competition": 4000,
        "trap": 3000,
    }
    assert generate.allocate_quota(15000) == {
        "chain": 2250,
        "merge": 3750,
        "alternative": 3750,
        "competition": 3000,
        "trap": 2250,
    }


def test_slots_follow_quota_and_are_stable():
    cfg = _cfg(count=100)
    slots = generate.structure_slots(cfg)
    assert len(slots) == 100
    counts = {name: slots.count(name) for name in generate.MAIN_STRUCTURES}
    assert counts == generate.allocate_quota(100)
    assert slots == generate.structure_slots(cfg)


def test_single_structure_slots():
    slots = generate.structure_slots(_cfg(structure="merge", count=9))
    assert slots == ["merge"] * 9


def test_trap_family_forces_trap_structure():
    slots = generate.structure_slots(_cfg(family="deep-trap", structure="mixed", count=5))
    assert slots == ["trap"] * 5
    with pytest.raises(ValueError):
        generate.validate_config(_cfg(family="deep-trap", structure="chain"))


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"rules": (3, 6), "steps": (2, 8)}, "--rules 下界"),
        ({"resources": (4, 6), "steps": (2, 8)}, "--resources 下界"),
        ({"distractors": 1.5}, "--distractors"),
        ({"arity": 0}, "--arity"),
        ({"klass": "both"}, "--class 需要 --pair"),
        ({"pair": ("forward-bfs", "forward-bfs"), "budget": 8}, "两个模式相同"),
        ({"pair": ("forward-bfs", "nope"), "budget": 8}, "未知模式"),
        ({"pair": ("forward-bfs", "forward-dfs")}, "--pair 需要 --budget"),
        ({"budget": 8}, "--budget 需要 --pair"),
    ],
)
def test_infeasible_configs_rejected(kwargs, reason):
    with pytest.raises(ValueError) as exc:
        generate.validate_config(_cfg(**kwargs))
    assert reason in str(exc.value)


def test_level_defaults_and_overrides():
    resolved = generate.resolve_level("hard", {"arity": 2, "resources": None})
    assert resolved["arity"] == 2
    assert resolved["resources"] == (24, 40)
    assert resolved["distractors"] == 0.55


def test_distractor_denominator_is_remaining_slots():
    """N_dis 以见证占用后的剩余槽位为分母，不是全部规则比例。"""
    cfg = _cfg(structure="chain", rules=(20, 20), steps=(4, 4), distractors=0.5, count=1)
    outcome = _first_candidate(cfg, "chain")
    budget = outcome.world.meta["rule_budget"]
    assert budget["rules"] == 20
    assert budget["witness"] == 4
    n_rest = budget["rules"] - budget["witness"]
    assert budget["distractor_target"] == int(n_rest * 0.5 + 0.5)
    assert budget["alternative_target"] == n_rest - budget["distractor_target"]


@pytest.mark.parametrize("structure", generate.MAIN_STRUCTURES)
def test_each_structure_produces_valid_evidence(structure):
    cfg = _cfg(structure=structure, level="medium", resources=(16, 24), initial_units=(4, 8),
               rules=(24, 48), steps=(4, 8), arity=2, distractors=0.4)
    made = 0
    for attempt in range(60):
        outcome = generate.make_candidate(cfg, structure, 0, attempt)
        if not isinstance(outcome, generate.Candidate):
            continue
        world = outcome.world
        env.validate_world(world)
        meta = world.meta
        assert meta["structure"] == structure
        assert env.verify_plan(world, meta["witness_plan"]).valid
        assert not env.meets_goal(world.initial, world.goal)
        evidence = meta["structure_evidence"]
        if structure == "chain":
            assert evidence["chain_depth"] >= 2
        elif structure == "merge":
            rule = world.rule(evidence["merge_rule"])
            assert rule is not None
            assert len(evidence["merge_inputs"]) >= 2
        elif structure == "alternative":
            assert evidence["alt_plans"]
            for alt in evidence["alt_plans"]:
                assert env.verify_plan(world, alt).valid
                assert set(alt) != set(meta["witness_plan"])
        elif structure == "competition":
            state = tuple(evidence["competition_state"])
            left, right = (world.rule(r) for r in evidence["competition_rules"])
            assert left is not None and right is not None
            assert env.applicable(state, left) and env.applicable(state, right)
            joint = tuple(a + b for a, b in zip(left.consume, right.consume))
            assert not env.dominates(state, joint)
            states = generate.witness_states(world, meta["witness_plan"])
            assert state in states
        elif structure == "trap":
            assert evidence["trap_entry_rules"]
            for rule_id in evidence["trap_entry_rules"]:
                assert world.rule(rule_id) is not None
        made += 1
        if made >= 3:
            break
    assert made >= 3, f"{structure} 接受率过低"


def test_candidate_is_deterministic_for_same_seed_and_slot():
    cfg = _cfg(structure="merge")
    a = generate.make_candidate(cfg, "merge", 3, 1)
    b = generate.make_candidate(cfg, "merge", 3, 1)
    assert type(a) is type(b)
    if isinstance(a, str):
        assert a == b
        return
    assert a.world.to_dict() == b.world.to_dict()
    assert a.fingerprint == b.fingerprint


def _first_candidate(cfg, structure, slot=0, tries=40):
    for attempt in range(tries):
        outcome = generate.make_candidate(cfg, structure, slot, attempt)
        if isinstance(outcome, generate.Candidate):
            return outcome
    raise AssertionError(f"{structure} 在 {tries} 次尝试内未产出候选")


def test_fingerprint_invariant_under_renaming_and_reordering():
    cfg = _cfg(structure="merge")
    outcome = _first_candidate(cfg, "merge")
    world = outcome.world
    # 符号重命名 + 规则重排后指纹不变，同构检查为真。
    n = world.n_resources
    order = list(range(n))[::-1]
    remap = {i: order[i] for i in range(n)}

    def permute(vec):
        out = [0] * n
        for i, v in enumerate(vec):
            out[remap[i]] = v
        return tuple(out)

    rules = [
        env.Rule(f"r{i}", permute(r.consume), permute(r.produce))
        for i, r in enumerate(reversed(world.rules))
    ]
    twin = env.World(
        symbols=tuple(f"Z{i}" for i in range(n)),
        initial=permute(world.initial),
        goal=permute(world.goal),
        rules=tuple(rules),
        world_id="w_twin",
        group_id=world.group_id,
    )
    env.validate_world(twin)
    assert env.structure_fingerprint(twin) == env.structure_fingerprint(world)
    verdict, _ = env.isomorphic(world, twin)
    assert verdict is True


def test_fingerprint_differs_for_different_structures():
    cfg = _cfg(structure="chain")
    seen = {}
    for attempt in range(12):
        outcome = generate.make_candidate(cfg, "chain", attempt, 0)
        if isinstance(outcome, generate.Candidate):
            seen[outcome.fingerprint] = outcome.world.world_id
    assert len(seen) >= 8


def test_qualification_records_cost_or_lower_bound():
    cfg = _cfg(
        family="deep-trap",
        structure="trap",
        pair=("forward-bfs", "forward-dfs"),
        budget=24,
        klass="any",
    )
    found = {}
    for attempt in range(80):
        outcome = generate.make_candidate(cfg, "trap", 0, attempt)
        if not isinstance(outcome, generate.Candidate):
            continue
        q = outcome.world.meta["qualification"]
        assert q["budget"] == 24
        assert q["class"] in ("both", "first-only", "second-only", "neither")
        for mode, cost in q["costs"].items():
            if cost["status"] == search.STATUS_SOLVED:
                assert cost["cost_exact"] is not None and cost["cost_exact"] <= 24
                assert cost["cost_lower_bound"] is None
            elif cost["status"] == search.STATUS_BUDGET:
                assert cost["cost_exact"] is None
                assert cost["cost_lower_bound"] == 24
        found[q["class"]] = True
        if len(found) >= 2:
            break
    assert found


def test_class_filter_rejects_other_classes():
    cfg = _cfg(
        family="deep-trap",
        structure="trap",
        pair=("forward-bfs", "forward-dfs"),
        budget=24,
        klass="first-only",
    )
    accepted = 0
    for attempt in range(120):
        outcome = generate.make_candidate(cfg, "trap", 0, attempt)
        if isinstance(outcome, generate.Candidate):
            assert outcome.world.meta["qualification"]["class"] == "first-only"
            accepted += 1
        else:
            assert isinstance(outcome, str)
    assert accepted >= 1

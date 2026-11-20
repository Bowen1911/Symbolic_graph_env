"""问题生成：family × structure × 复杂度参数，植入见证方案并做结构去重。

生成流程按第 6.1 节：构造 → 替代/竞争 → 干扰 → 打乱符号与规则 ID → 独立执行器验证
→ 可选的策略优势筛选 → 去重写入。槽位分配按第 6.3 节公式，配额按第 6.4 节。
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from . import env, search
from .env import Rule, World

FAMILIES = ("foundation", "general", "deep-trap", "wide-trap")
STRUCTURES = ("chain", "merge", "alternative", "competition", "trap", "divergence", "mixed")
MAIN_STRUCTURES = ("chain", "merge", "alternative", "competition", "trap")
TRAP_FAMILIES = ("deep-trap", "wide-trap")

# 第 6.4 节 mixed 默认配额，版本化常量。
MIXED_QUOTA: dict[str, float] = {
    "chain": 0.15,
    "merge": 0.25,
    "alternative": 0.25,
    "competition": 0.20,
    "trap": 0.15,
}
QUOTA_VERSION = "1.0"

# 第 6.3 节 level 预设。
LEVELS: dict[str, dict] = {
    "easy": {
        "resources": (8, 12),
        "initial_units": (3, 5),
        "rules": (10, 18),
        "steps": (2, 4),
        "arity": 2,
        "distractors": 0.25,
    },
    "medium": {
        "resources": (16, 24),
        "initial_units": (4, 8),
        "rules": (24, 48),
        "steps": (4, 8),
        "arity": 2,
        "distractors": 0.40,
    },
    "hard": {
        "resources": (24, 40),
        "initial_units": (6, 10),
        "rules": (48, 80),
        "steps": (8, 12),
        "arity": 3,
        "distractors": 0.55,
    },
}

CLASSES = ("both", "first-only", "second-only", "neither", "any")

Range = tuple[int, int]


def parse_range(spec: str, name: str) -> Range:
    """解析正整数或闭区间 `lo:hi`。"""
    text = str(spec).strip()
    if ":" in text:
        lo_s, hi_s = text.split(":", 1)
        lo, hi = int(lo_s), int(hi_s)
    else:
        lo = hi = int(text)
    if lo < 1 or hi < lo:
        raise ValueError(f"{name} 区间非法: {spec}")
    return lo, hi


@dataclass
class GenConfig:
    family: str = "general"
    structure: str = "mixed"
    level: str | None = None
    resources: Range = (16, 24)
    initial_units: Range = (4, 8)
    rules: Range = (24, 48)
    steps: Range = (4, 8)
    arity: int = 2
    distractors: float = 0.40
    count: int = 100
    seed: int = 0
    pair: tuple[str, str] | None = None
    budget: int | None = None
    klass: str = "any"
    max_attempts: int = 0
    max_states: int = search.DEFAULT_MAX_STATES
    forced_prefix: int | None = None
    shared_prefix: int | None = None
    divergence_expansions: Range | None = None
    plan_shared_prefix: Range | None = None
    critical_steps: Range | None = None
    criticality_states: int = env.DEFAULT_CRITICALITY_STATES

    def to_dict(self) -> dict:
        return {
            "generator_version": env.GENERATOR_VERSION,
            "quota_version": QUOTA_VERSION,
            "family": self.family,
            "structure": self.structure,
            "level": self.level,
            "resources": list(self.resources),
            "initial_units": list(self.initial_units),
            "rules": list(self.rules),
            "steps": list(self.steps),
            "arity": self.arity,
            "distractors": self.distractors,
            "count": self.count,
            "seed": self.seed,
            "pair": list(self.pair) if self.pair else None,
            "budget": self.budget,
            "class": self.klass,
            "max_attempts": self.effective_max_attempts,
            "max_states": self.max_states,
            "forced_prefix": self.forced_prefix,
            "shared_prefix": self.shared_prefix,
            "divergence_expansions": list(self.divergence_expansions)
            if self.divergence_expansions
            else None,
            "plan_shared_prefix": list(self.plan_shared_prefix)
            if self.plan_shared_prefix
            else None,
            "critical_steps": list(self.critical_steps) if self.critical_steps else None,
            "criticality_states": self.criticality_states,
        }

    @property
    def structural_filters(self) -> bool:
        """是否启用了第 7.1 节的分歧位置或关键步筛选。"""
        return any(
            (self.divergence_expansions, self.plan_shared_prefix, self.critical_steps)
        )

    @property
    def effective_max_attempts(self) -> int:
        return self.max_attempts or 50 * self.count


def validate_config(cfg: GenConfig) -> None:
    if cfg.family not in FAMILIES:
        raise ValueError(f"未知 --family: {cfg.family}；允许 {','.join(FAMILIES)}")
    if cfg.structure not in STRUCTURES:
        raise ValueError(f"未知 --structure: {cfg.structure}；允许 {','.join(STRUCTURES)}")
    if cfg.family in TRAP_FAMILIES and cfg.structure not in ("trap", "mixed"):
        raise ValueError(f"{cfg.family} 只允许 --structure trap")
    if not 0.0 <= cfg.distractors <= 1.0:
        raise ValueError("--distractors 应在 0 到 1 之间")
    if cfg.arity < 1:
        raise ValueError("--arity 应为正整数")
    if cfg.klass not in CLASSES:
        raise ValueError(f"未知 --class: {cfg.klass}；允许 {','.join(CLASSES)}")
    if cfg.klass != "any" and cfg.pair is None:
        raise ValueError("--class 需要 --pair")
    if cfg.budget is not None and cfg.pair is None:
        raise ValueError("--budget 需要 --pair")
    if cfg.pair is not None:
        if cfg.budget is None:
            raise ValueError("--pair 需要 --budget")
        if cfg.pair[0] == cfg.pair[1]:
            raise ValueError("--pair 的两个模式相同")
        for mode in cfg.pair:
            if mode not in search.MODES:
                raise ValueError(f"--pair 含未知模式: {mode}")
    if cfg.forced_prefix is not None:
        if cfg.structure != "chain":
            raise ValueError("--forced-prefix 要求 --structure chain")
        if cfg.forced_prefix < 1 or cfg.forced_prefix >= cfg.steps[0]:
            raise ValueError(
                f"--forced-prefix 应满足 1 <= k < --steps 下界 {cfg.steps[0]}"
            )
    if cfg.shared_prefix is not None:
        if cfg.structure != "divergence":
            raise ValueError("--shared-prefix 要求 --structure divergence")
        if cfg.shared_prefix < 1 or cfg.shared_prefix > cfg.steps[0] - 2:
            raise ValueError(
                f"--shared-prefix 应满足 1 <= j <= --steps 下界 - 2 = {cfg.steps[0] - 2}"
            )
    if cfg.structure == "divergence" and cfg.steps[0] < 3:
        raise ValueError("--structure divergence 要求 --steps 下界不小于 3")
    if cfg.structural_filters and cfg.pair is None:
        raise ValueError("--divergence-expansions/--plan-shared-prefix/--critical-steps 需要 --pair")
    # 规则数下界必须容纳见证方案所需的不同规则数（第 6.3 节）。
    if cfg.rules[0] < cfg.steps[1]:
        raise ValueError(
            f"--rules 下界 {cfg.rules[0]} 小于 --steps 上界 {cfg.steps[1]}，见证方案无法放入"
        )
    # 资源数下界需容纳目标、中间对象与至少一个基础资源。
    if cfg.resources[0] < cfg.steps[1] + 1:
        raise ValueError(
            f"--resources 下界 {cfg.resources[0]} 不足以容纳 {cfg.steps[1]} 步见证方案"
        )


def resolve_level(level: str | None, overrides: dict) -> dict:
    """`--level` 提供默认值，显式参数覆盖 level。"""
    base = dict(LEVELS[level]) if level else {}
    out = dict(LEVELS["medium"])
    out.update(base)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def apply_family(cfg: GenConfig) -> GenConfig:
    """family 对复杂度参数的修正（第 6.2 节）。"""
    if cfg.family == "foundation":
        lo, hi = cfg.steps
        cfg = _replace(cfg, steps=(lo, min(hi, max(lo, 4))), distractors=cfg.distractors * 0.5)
    elif cfg.family == "wide-trap":
        lo, hi = cfg.steps
        cfg = _replace(cfg, steps=(max(lo, (lo + hi + 1) // 2), hi))
    return cfg


def _replace(cfg: GenConfig, **kwargs) -> GenConfig:
    data = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
    data.update(kwargs)
    return GenConfig(**data)


# --------------------------------------------------------------------------- #
# 配额与槽位
# --------------------------------------------------------------------------- #


def allocate_quota(count: int, quota: dict[str, float] = MIXED_QUOTA) -> dict[str, int]:
    """最大余数法，平局按 MAIN_STRUCTURES 顺序打破。"""
    raw = {k: count * quota[k] for k in MAIN_STRUCTURES}
    base = {k: int(raw[k]) for k in MAIN_STRUCTURES}
    remainder = count - sum(base.values())
    order = sorted(
        MAIN_STRUCTURES, key=lambda k: (-(raw[k] - base[k]), MAIN_STRUCTURES.index(k))
    )
    for k in order[:remainder]:
        base[k] += 1
    return base


def structure_slots(cfg: GenConfig) -> list[str]:
    """预定结构槽位。同一 seed 下与 workers 无关。"""
    if cfg.structure != "mixed":
        return [cfg.structure] * cfg.count
    if cfg.family in TRAP_FAMILIES:
        return ["trap"] * cfg.count
    quota = allocate_quota(cfg.count)
    slots: list[str] = []
    for name in MAIN_STRUCTURES:
        slots.extend([name] * quota[name])
    random.Random(_derive_seed(cfg.seed, "slots")).shuffle(slots)
    return slots


def _derive_seed(*parts) -> int:
    payload = ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


# --------------------------------------------------------------------------- #
# 候选构造
# --------------------------------------------------------------------------- #


class Reject(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class ConsRule:
    tag: str
    consume: dict[str, int]
    produce: dict[str, int]
    role: str  # witness / alternative / distractor / competition


@dataclass
class Construction:
    roles: list[str]
    initial: dict[str, int]
    goal: dict[str, int]
    rules: list[ConsRule]
    witness_tags: list[str]
    alt_tags: list[list[str]]
    tree_children: list[list[int]]
    structure: str
    rule_budget: dict[str, int]
    params: dict


def symbol_names(n: int) -> list[str]:
    letters = string.ascii_uppercase
    out: list[str] = []
    i = 0
    while len(out) < n:
        if i < 26:
            out.append(letters[i])
        else:
            out.append(letters[i // 26 - 1] + letters[i % 26])
        i += 1
    return out


def _build_tree(rng: random.Random, n_nodes: int, arity: int, structure: str) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(n_nodes)]
    start = 1
    if structure == "merge":
        if n_nodes < 3 or arity < 2:
            raise Reject("merge_needs_three_steps")
        children[0] = [1, 2]
        start = 3
    for i in range(start, n_nodes):
        if structure in ("chain", "divergence"):
            parent = i - 1
        else:
            cand = [p for p in range(i) if len(children[p]) < arity]
            if not cand:
                raise Reject("tree_arity_exhausted")
            parent = rng.choice(cand)
        children[parent].append(i)
    if structure in ("chain", "divergence") and any(len(c) > 1 for c in children):
        raise Reject("chain_not_path")
    return children


def _tree_height(children: Sequence[Sequence[int]], node: int = 0) -> int:
    if not children[node]:
        return 1
    return 1 + max(_tree_height(children, c) for c in children[node])


def _subtree_size(children: Sequence[Sequence[int]], node: int) -> int:
    return 1 + sum(_subtree_size(children, c) for c in children[node])


def _postorder(children: Sequence[Sequence[int]], node: int = 0) -> list[int]:
    out: list[int] = []
    for c in children[node]:
        out.extend(_postorder(children, c))
    out.append(node)
    return out


def build_construction(cfg: GenConfig, structure: str, rng: random.Random) -> Construction:
    n_res = rng.randint(*cfg.resources)
    target_units = rng.randint(*cfg.initial_units)
    n_rules = rng.randint(*cfg.rules)
    n_steps = rng.randint(*cfg.steps)
    arity = cfg.arity

    n_rest = n_rules - n_steps
    if n_rest < 0:
        raise Reject("rule_budget_infeasible")
    n_dis = int(n_rest * cfg.distractors + 0.5)
    n_alt = n_rest - n_dis

    children = _build_tree(rng, n_steps, arity, structure)

    # 叶子槽位：每个无子节点至少一个，再按剩余初始资源预算分配。
    leaf = [1 if not children[i] else 0 for i in range(n_steps)]
    budget = target_units - sum(leaf)
    if budget < 0:
        raise Reject("initial_units_too_small")
    order = list(range(n_steps))
    rng.shuffle(order)
    for i in order:
        room = arity - (len(children[i]) + leaf[i])
        while room > 0 and budget > 0 and rng.random() < 0.55:
            leaf[i] += 1
            budget -= 1
            room -= 1
    total_leaf = sum(leaf)

    # 符号角色分配。
    n_alt_steps = 0
    shared_prefix = None
    if structure == "alternative":
        n_alt_steps = min(rng.randint(*cfg.steps), n_alt)
        if n_alt_steps < 2:
            raise Reject("alternative_slots_too_few")
    elif structure == "divergence":
        if n_steps < 3:
            raise Reject("divergence_needs_three_steps")
        shared_prefix = (
            cfg.shared_prefix
            if cfg.shared_prefix is not None
            else rng.randint(1, n_steps - 2)
        )
        if not 1 <= shared_prefix <= n_steps - 2:
            raise Reject("shared_prefix_out_of_range")
        n_alt_steps = n_steps - shared_prefix
        if n_alt_steps > n_alt:
            raise Reject("divergence_slots_too_few")
    avail = n_res - n_steps - n_alt_steps  # 目标 + 见证中间 + 替代中间 已占 n_steps+n_alt_steps
    if avail < 1:
        raise Reject("resources_too_few")
    n_base = min(max(1, total_leaf), max(1, avail // 2))
    n_decoy = avail - n_base
    if n_decoy < 0:
        raise Reject("resources_too_few")

    goal_role = "G"
    inter_roles = [f"I{i}" for i in range(1, n_steps)]
    alt_roles = [f"J{i}" for i in range(1, n_alt_steps)] if n_alt_steps else []
    base_roles = [f"B{i}" for i in range(n_base)]
    decoy_roles = [f"D{i}" for i in range(n_decoy)]
    roles = [goal_role] + inter_roles + alt_roles + base_roles + decoy_roles
    if len(roles) != n_res:
        # 余下的符号作为未被任何规则使用的无关资源。
        decoy_roles.extend(f"D{n_decoy + k}" for k in range(n_res - len(roles)))
        n_decoy = len(decoy_roles)
        roles = [goal_role] + inter_roles + alt_roles + base_roles + decoy_roles

    out_role = {0: goal_role}
    for idx, node in enumerate(range(1, n_steps)):
        out_role[node] = inter_roles[idx]

    initial: dict[str, int] = {}
    witness_rules: list[ConsRule] = []
    for node in range(n_steps):
        consume: dict[str, int] = {}
        for child in children[node]:
            consume[out_role[child]] = consume.get(out_role[child], 0) + 1
        for _ in range(leaf[node]):
            sym = rng.choice(base_roles)
            consume[sym] = consume.get(sym, 0) + 1
            initial[sym] = initial.get(sym, 0) + 1
        witness_rules.append(
            ConsRule(tag=f"w{node}", consume=consume, produce={out_role[node]: 1}, role="witness")
        )

    # 把初始资源补到 target_units：优先 decoy，其次 base，不加入目标资源。
    # forced_prefix 下只用 decoy，避免基础资源的额外单位使见证规则在前段重复可执行。
    pad = target_units - sum(initial.values())
    if cfg.forced_prefix is not None:
        pad_pool = decoy_roles
    else:
        pad_pool = (decoy_roles + base_roles) if decoy_roles else base_roles
    for _ in range(max(0, pad)):
        if not pad_pool:
            break
        sym = rng.choice(pad_pool)
        initial[sym] = initial.get(sym, 0) + 1

    rules = list(witness_rules)
    alt_tags: list[list[str]] = []
    used_alt = 0

    if structure == "alternative":
        alt_children = _build_tree(rng, n_alt_steps, arity, "alternative")
        alt_out = {0: goal_role}
        for idx, node in enumerate(range(1, n_alt_steps)):
            alt_out[node] = alt_roles[idx]
        remaining = dict(initial)
        alt_rules: list[ConsRule] = []
        for node in range(n_alt_steps):
            consume: dict[str, int] = {}
            for child in alt_children[node]:
                consume[alt_out[child]] = consume.get(alt_out[child], 0) + 1
            n_leaf = 1 if not alt_children[node] else 0
            for _ in range(n_leaf):
                pool = [s for s in base_roles if remaining.get(s, 0) > 0]
                if not pool:
                    raise Reject("alternative_leaf_unavailable")
                sym = rng.choice(pool)
                consume[sym] = consume.get(sym, 0) + 1
                remaining[sym] -= 1
            alt_rules.append(
                ConsRule(tag=f"a{node}", consume=consume, produce={alt_out[node]: 1}, role="alternative")
            )
        rules.extend(alt_rules)
        used_alt = len(alt_rules)
        alt_tags.append([f"a{n}" for n in _postorder(alt_children)])

    if structure == "divergence":
        # 见证方案的前 j 步后状态上接出一条使用新中间对象的尾段。
        assert shared_prefix is not None
        j = shared_prefix
        m = n_alt_steps
        branch_role = out_role[n_steps - j]
        div_rules: list[ConsRule] = []
        for i in range(m):
            src = branch_role if i == 0 else alt_roles[i - 1]
            dst = goal_role if i == m - 1 else alt_roles[i]
            if src == dst:
                raise Reject("divergence_degenerate_rule")
            div_rules.append(
                ConsRule(tag=f"a{i}", consume={src: 1}, produce={dst: 1}, role="alternative")
            )
        rules.extend(div_rules)
        used_alt = m
        witness_order = [f"w{n}" for n in _postorder(children)]
        alt_tags.append(witness_order[:j] + [f"a{i}" for i in range(m)])

    # 竞争规则：在见证路径上制造两个分别可执行但不能同时满足的规则。
    if structure == "competition":
        if not decoy_roles:
            raise Reject("competition_needs_decoy")
        singles = [s for s in base_roles if initial.get(s, 0) == 1]
        target = None
        for cons in witness_rules:
            for sym in cons.consume:
                if sym in singles:
                    target = sym
                    break
            if target:
                break
        if target is None:
            raise Reject("competition_no_single_base")
        rules.append(
            ConsRule(
                tag="c0",
                consume={target: 1},
                produce={rng.choice(decoy_roles): 1},
                role="competition",
            )
        )
        n_dis = max(0, n_dis - 1)

    # 替代有效构造：为已有中间对象增加另一条产生方式。
    # 用见证树的子树规模约束输入深度，否则替代规则会形成绕过见证链的捷径，
    # 使 --steps 对实际解长度失去控制。
    existing = {(_freeze(r.consume), _freeze(r.produce)) for r in rules}
    inter_pool = inter_roles + alt_roles
    sub_by_role: dict[str, int] = {
        out_role[node]: _subtree_size(children, node) for node in range(n_steps)
    }
    filled = 0
    want_alt = max(0, n_alt - used_alt)
    for _ in range(want_alt * 8):
        if filled >= want_alt:
            break
        target = rng.choice(inter_pool) if inter_pool else goal_role
        if rng.random() < 0.3:
            target = goal_role
        pool = [s for s in base_roles + inter_pool + decoy_roles if s != target]
        if not pool:
            break
        size = rng.randint(1, arity)
        consume: dict[str, int] = {}
        for _ in range(size):
            sym = rng.choice(pool)
            consume[sym] = consume.get(sym, 0) + 1
        depths = [sub_by_role[s] for s in consume if s in sub_by_role]
        if not depths or max(depths) < sub_by_role.get(target, 1) - 1:
            continue
        key = (_freeze(consume), _freeze({target: 1}))
        if key in existing:
            continue
        existing.add(key)
        rules.append(
            ConsRule(tag=f"v{filled}", consume=consume, produce={target: 1}, role="alternative")
        )
        filled += 1
    used_alt += filled

    # 干扰构造：deep-trap 用链式，wide-trap 用并列单步，其余混合。
    trap_entry_tags: list[str] = []
    placed_dis = 0
    if n_dis > 0:
        layout = (
            "chain"
            if cfg.family == "deep-trap"
            else "wide"
            if cfg.family == "wide-trap"
            else ("chain" if rng.random() < 0.5 else "wide")
        )
        entry_pool = [s for s in initial if initial[s] > 0] + inter_roles
        if not entry_pool or not decoy_roles:
            n_dis = 0
        chain_prev: str | None = None
        for _ in range(n_dis * 8):
            if placed_dis >= n_dis:
                break
            if layout == "chain" and chain_prev is not None:
                src = chain_prev
            else:
                src = rng.choice(entry_pool)
            dst = rng.choice(decoy_roles)
            if dst == src:
                continue
            consume = {src: 1}
            if arity >= 2 and rng.random() < 0.3:
                extra = rng.choice(decoy_roles + base_roles)
                if extra != src:
                    consume[extra] = consume.get(extra, 0) + 1
            key = (_freeze(consume), _freeze({dst: 1}))
            if key in existing:
                continue
            existing.add(key)
            tag = f"d{placed_dis}"
            rules.append(ConsRule(tag=tag, consume=consume, produce={dst: 1}, role="distractor"))
            if chain_prev is None or layout == "wide":
                trap_entry_tags.append(tag)
            chain_prev = dst if layout == "chain" else None
            placed_dis += 1

    if cfg.forced_prefix is not None:
        gate = out_role[n_steps - cfg.forced_prefix]
        gated: list[ConsRule] = []
        for cons_rule in rules:
            if cons_rule.role == "witness":
                gated.append(cons_rule)
                continue
            if gate in cons_rule.produce:
                continue
            consume = dict(cons_rule.consume)
            consume.pop(gate, None)
            while sum(consume.values()) > max(0, arity - 1):
                consume.pop(next(iter(consume)))
            consume[gate] = 1
            if consume == cons_rule.produce:
                continue
            gated.append(
                ConsRule(
                    tag=cons_rule.tag,
                    consume=consume,
                    produce=cons_rule.produce,
                    role=cons_rule.role,
                )
            )
        # 去掉注入后重复的规则
        seen: set[tuple] = set()
        rules = []
        for cons_rule in gated:
            key = (_freeze(cons_rule.consume), _freeze(cons_rule.produce))
            if key in seen:
                continue
            seen.add(key)
            rules.append(cons_rule)

    witness_tags = [f"w{n}" for n in _postorder(children)]
    return Construction(
        roles=roles,
        initial=initial,
        goal={goal_role: 1},
        rules=rules,
        witness_tags=witness_tags,
        alt_tags=alt_tags,
        tree_children=children,
        structure=structure,
        rule_budget={
            "rules": n_rules,
            "witness": n_steps,
            "distractor_target": n_dis,
            "distractor_actual": placed_dis,
            "alternative_target": n_alt,
            "alternative_actual": used_alt,
            "actual": len(rules),
        },
        params={
            "resources": n_res,
            "initial_units": sum(initial.values()),
            "initial_units_target": target_units,
            "shared_prefix": shared_prefix,
            "forced_prefix": cfg.forced_prefix,
            "arity": arity,
            "steps": n_steps,
            "distractors": cfg.distractors,
            "trap_entry_tags": trap_entry_tags,
        },
    )


def _freeze(mapping: dict[str, int]) -> tuple:
    return tuple(sorted(mapping.items()))


def construction_payload(cons: Construction) -> str:
    """group_id 的来源：角色名命名的底层构造，与最终符号和规则顺序无关。"""
    return json.dumps(
        {
            "structure": cons.structure,
            "initial": sorted(cons.initial.items()),
            "goal": sorted(cons.goal.items()),
            "rules": sorted(
                [sorted(r.consume.items()), sorted(r.produce.items())] for r in cons.rules
            ),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


# --------------------------------------------------------------------------- #
# 构造 -> World
# --------------------------------------------------------------------------- #


def materialize(cons: Construction, rng: random.Random) -> tuple[World, dict[str, str]]:
    """打乱符号与规则展示顺序，重新分配规则 ID（第 6.1 节第 4 步）。"""
    names = symbol_names(len(cons.roles))
    perm = list(cons.roles)
    rng.shuffle(perm)
    role_to_symbol = {role: names[i] for i, role in enumerate(perm)}
    symbols = tuple(names)
    index = {name: i for i, name in enumerate(symbols)}

    def vec(mapping: dict[str, int]) -> tuple[int, ...]:
        out = [0] * len(symbols)
        for role, count in mapping.items():
            out[index[role_to_symbol[role]]] = count
        return tuple(out)

    order = list(range(len(cons.rules)))
    rng.shuffle(order)
    tag_to_id: dict[str, str] = {}
    rules: list[Rule] = []
    for new_idx, old_idx in enumerate(order):
        cons_rule = cons.rules[old_idx]
        rule_id = f"r{new_idx}"
        tag_to_id[cons_rule.tag] = rule_id
        rules.append(Rule(id=rule_id, consume=vec(cons_rule.consume), produce=vec(cons_rule.produce)))

    initial = vec(cons.initial)
    goal = vec(cons.goal)
    world_id = env.make_world_id(symbols, initial, goal, rules)
    group_id = env.make_group_id(construction_payload(cons))
    world = World(
        symbols=symbols,
        initial=initial,
        goal=goal,
        rules=tuple(rules),
        world_id=world_id,
        group_id=group_id,
        meta={},
    )
    env.validate_world(world)
    return world, tag_to_id


# --------------------------------------------------------------------------- #
# 结构证据（第 6.4 节）
# --------------------------------------------------------------------------- #


def witness_states(world: World, plan: Sequence[str]) -> list[tuple[int, ...]]:
    states = [world.initial]
    state = world.initial
    for rule_id in plan:
        rule = world.rule(rule_id)
        assert rule is not None
        nxt = env.step(state, rule)
        if nxt is None:
            raise Reject("witness_not_executable")
        state = nxt
        states.append(state)
    return states


def find_competition(world: World, states: Sequence[tuple[int, ...]]):
    """在见证路径状态上枚举规则对，返回 (状态, 规则对)。"""
    n = world.n_resources
    for state in states:
        ready = [r for r in world.rules if env.applicable(state, r)]
        for i, ra in enumerate(ready):
            for rb in ready[i + 1 :]:
                joint = tuple(ra.consume[k] + rb.consume[k] for k in range(n))
                if not env.dominates(state, joint):
                    return state, (ra.id, rb.id)
    return None


def structure_evidence(
    world: World, cons: Construction, tag_to_id: dict[str, str], plan: Sequence[str]
) -> dict:
    structure = cons.structure
    states = witness_states(world, plan)
    if structure == "chain":
        return {"chain_depth": _tree_height(cons.tree_children)}
    if structure == "merge":
        for node, kids in enumerate(cons.tree_children):
            if len(kids) >= 2:
                rule_id = tag_to_id[f"w{node}"]
                rule = world.rule(rule_id)
                assert rule is not None
                inputs = [
                    world.symbols[i] for i, v in enumerate(rule.consume) if v
                ]
                return {"merge_rule": rule_id, "merge_inputs": inputs}
        raise Reject("merge_absent")
    if structure == "alternative":
        if not cons.alt_tags:
            raise Reject("alternative_absent")
        alt_plans = [[tag_to_id[t] for t in tags] for tags in cons.alt_tags]
        for alt in alt_plans:
            check = env.verify_plan(world, alt)
            if not check.valid:
                raise Reject("alternative_plan_invalid")
            if set(alt) == set(plan):
                raise Reject("alternative_same_rules")
        return {"alt_plans": alt_plans}
    if structure == "competition":
        found = find_competition(world, states)
        if found is None:
            raise Reject("competition_absent")
        state, pair = found
        return {"competition_state": list(state), "competition_rules": list(pair)}
    if structure == "divergence":
        if not cons.alt_tags:
            raise Reject("divergence_absent")
        alt = [tag_to_id[t] for t in cons.alt_tags[0]]
        if not env.verify_plan(world, alt).valid:
            raise Reject("divergence_plan_invalid")
        target = cons.params.get("shared_prefix")
        actual = search.common_prefix_length(plan, alt)
        if target is not None and actual != target:
            raise Reject("divergence_prefix_mismatch")
        if actual >= min(len(plan), len(alt)):
            raise Reject("divergence_no_tail")
        return {"alt_plans": [alt], "plan_shared_prefix": actual}
    if structure == "trap":
        tags = cons.params.get("trap_entry_tags") or []
        entries = [tag_to_id[t] for t in tags if t in tag_to_id]
        if not entries:
            raise Reject("trap_absent")
        return {"trap_entry_rules": entries}
    raise Reject(f"unknown_structure_{structure}")


def forced_prefix_holds(world: World, plan: Sequence[str], k: int) -> bool:
    """前 k 个状态上是否只有一条规则可执行（第 6.8 节）。"""
    state = world.initial
    for t in range(min(k, len(plan))):
        if sum(1 for r in world.rules if env.applicable(state, r)) != 1:
            return False
        rule = world.rule(plan[t])
        if rule is None:
            return False
        nxt = env.step(state, rule)
        if nxt is None:
            return False
        state = nxt
    return True


def structure_tags(world: World, cons: Construction, plan: Sequence[str]) -> list[str]:
    """主结构之外的辅助标签，按实际构造判断。"""
    tags = [cons.structure]
    if cons.structure != "chain" and _tree_height(cons.tree_children) >= 3:
        tags.append("chain")
    if cons.structure != "merge" and any(len(k) >= 2 for k in cons.tree_children):
        tags.append("merge")
    if cons.structure != "competition":
        states = witness_states(world, plan)
        if find_competition(world, states) is not None:
            tags.append("competition")
    return list(dict.fromkeys(tags))


# --------------------------------------------------------------------------- #
# 优势筛选（第 7 节）
# --------------------------------------------------------------------------- #


def qualify(world: World, cfg: GenConfig) -> dict | None:
    """按 pair/budget/class 认证优势类别。不满足返回 None。"""
    if cfg.pair is None:
        return None
    budget = int(cfg.budget or 0)
    limits = search.Limits(max_expansions=budget, max_states=cfg.max_states)
    outcomes = []
    costs = {}
    results: dict[str, search.SolveResult] = {}
    for mode in cfg.pair:
        result = search.solve(world, mode, limits)
        results[mode] = result
        if result.status == search.STATUS_ERROR:
            raise Reject("solver_error")
        within = search.cost_within_budget(result, budget)
        outcomes.append(within)
        costs[mode] = {
            "status": result.status,
            "cost_exact": result.cost.checked_nodes if result.status == search.STATUS_SOLVED else None,
            "cost_lower_bound": budget if result.status == search.STATUS_BUDGET else None,
        }
    if any(o is None for o in outcomes):
        raise Reject("qualification_unknown")
    first, second = outcomes
    actual = (
        "both"
        if first and second
        else "first-only"
        if first and not second
        else "second-only"
        if second and not first
        else "neither"
    )
    if cfg.klass != "any" and actual != cfg.klass:
        raise Reject(f"class_{actual}")

    structural = None
    if cfg.structural_filters:
        left, right = (results[m] for m in cfg.pair)
        if not (left.solved and right.solved):
            raise Reject("divergence_unknown")
        record = search.divergence_record(world.world_id, cfg.pair, left, right)
        structural = {
            "shared_lines": record["shared_lines"],
            "shared_expansions": record["shared_expansions"],
            "plan_shared_prefix": record["plan_shared_prefix"],
            "critical_steps": None,
        }
        if cfg.divergence_expansions is not None:
            value = record["shared_expansions"]
            lo, hi = cfg.divergence_expansions
            if not lo <= value <= hi:
                raise Reject(f"divergence_expansions_{value}")
        if cfg.plan_shared_prefix is not None:
            value = record["plan_shared_prefix"]
            lo, hi = cfg.plan_shared_prefix
            if not lo <= value <= hi:
                raise Reject(f"plan_shared_prefix_{value}")
        if cfg.critical_steps is not None:
            crit = env.plan_outcome(world, left.plan or [], cfg.criticality_states)
            if crit.unknown_steps:
                raise Reject("criticality_unknown")
            value = crit.critical_steps
            structural["critical_steps"] = value
            lo, hi = cfg.critical_steps
            if not lo <= value <= hi:
                raise Reject(f"critical_steps_{value}")

    return {
        "pair": list(cfg.pair),
        "budget": budget,
        "class": actual,
        "costs": costs,
        "structural": structural,
    }


# --------------------------------------------------------------------------- #
# 单个候选
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    world: World
    fingerprint: str
    rejection: str | None = None


def make_candidate(cfg: GenConfig, structure: str, slot: int, attempt: int) -> Candidate | str:
    """生成一个候选并做本地校验。返回 Candidate 或拒绝原因。"""
    rng = random.Random(_derive_seed(cfg.seed, slot, attempt))
    try:
        cons = build_construction(cfg, structure, rng)
        world, tag_to_id = materialize(cons, rng)
        plan = [tag_to_id[t] for t in cons.witness_tags]
        check = env.verify_plan(world, plan)
        if not check.valid:
            return "witness_not_executable"
        if env.meets_goal(world.initial, world.goal):
            return "initial_already_goal"
        evidence = structure_evidence(world, cons, tag_to_id, plan)
        if cfg.forced_prefix is not None and not forced_prefix_holds(
            world, plan, cfg.forced_prefix
        ):
            return "forced_prefix_infeasible"
        tags = structure_tags(world, cons, plan)
        qualification = qualify(world, cfg)
        meta = {
            "family": cfg.family,
            "structure": structure,
            "structure_tags": tags,
            "structure_evidence": evidence,
            "rule_budget": cons.rule_budget,
            "params": {k: v for k, v in cons.params.items() if k != "trap_entry_tags"},
            "seed": cfg.seed,
            "candidate_index": slot,
            "attempt": attempt,
            "witness_plan": plan,
            "witness_steps": len(plan),
            "qualification": qualification,
        }
        world = World(
            symbols=world.symbols,
            initial=world.initial,
            goal=world.goal,
            rules=world.rules,
            world_id=world.world_id,
            group_id=world.group_id,
            meta=meta,
        )
        return Candidate(world=world, fingerprint=env.structure_fingerprint(world))
    except Reject as exc:
        return exc.reason
    except env.WorldError as exc:
        return f"invalid_world:{exc}"


def slot_worker(args: tuple) -> tuple:
    """进程池入口：对单个槽位尝试若干候选，返回首个本地通过的候选。"""
    cfg, structure, slot, attempt_lo, attempt_hi = args
    rejections: Counter = Counter()
    for attempt in range(attempt_lo, attempt_hi):
        outcome = make_candidate(cfg, structure, slot, attempt)
        if isinstance(outcome, Candidate):
            return slot, outcome, attempt - attempt_lo + 1, rejections
        rejections[outcome] += 1
    return slot, None, attempt_hi - attempt_lo, rejections

"""四个子命令的命令行入口：generate / solve / export / verify。

统一入口 `python -m symplan`。所有命令的参数以 `--help` 展示；命令行参数覆盖
`--level` 预设。每个生产命令在输出旁写出机器可读统计（第 6.6 节）。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Sequence
from pathlib import Path

from . import env, generate, qa, search, verify
from .env import World
from .search import SolveResult

RESULT_ROOT = Path("result")


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #


class Progress:
    """终端与日志同时输出的进度，含 resume 起点与 ETA（跳过 resume 部分的估计）。"""

    def __init__(self, total: int, label: str, *, done: int = 0, stream=sys.stderr):
        self.total = total
        self.label = label
        self.resumed = done
        self.done = done
        self.stream = stream
        self.start = time.monotonic()
        self.last_emit = 0.0

    def advance(self, n: int = 1, extra: str = "") -> None:
        self.done += n
        now = time.monotonic()
        if self.done < self.total and now - self.last_emit < 0.5:
            return
        self.last_emit = now
        fresh = self.done - self.resumed
        elapsed = now - self.start
        rate = fresh / elapsed if elapsed > 0 and fresh > 0 else 0.0
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 else float("nan")
        pct = 100.0 * self.done / self.total if self.total else 100.0
        msg = (
            f"[{self.label}] {self.done}/{self.total} ({pct:.1f}%) "
            f"resume={self.resumed} rate={rate:.1f}/s eta={_fmt_eta(eta)}"
        )
        if extra:
            msg += " " + extra
        print(msg, file=self.stream, flush=True)

    def close(self) -> None:
        self.advance(0)


def _fmt_eta(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "n/a"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


def quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])

        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return float(ordered[idx])

    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "p50": q(0.5),
        "p90": q(0.9),
        "p99": q(0.99),
        "max": float(ordered[-1]),
        "mean": round(statistics.fmean(ordered), 3),
    }


def file_sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def cmd_generate(args: argparse.Namespace) -> int:
    overrides = {
        "resources": generate.parse_range(args.resources, "--resources") if args.resources else None,
        "initial_units": generate.parse_range(args.initial_units, "--initial-units")
        if args.initial_units
        else None,
        "rules": generate.parse_range(args.rules, "--rules") if args.rules else None,
        "steps": generate.parse_range(args.steps, "--steps") if args.steps else None,
        "arity": args.arity,
        "distractors": args.distractors,
    }
    resolved = generate.resolve_level(args.level, overrides)
    pair = tuple(m.strip() for m in args.pair.split(",")) if args.pair else None
    if pair is not None and len(pair) != 2:
        raise SystemExit("--pair 需要两个以逗号分隔的模式")
    cfg = generate.GenConfig(
        family=args.family,
        structure=args.structure,
        level=args.level,
        resources=resolved["resources"],
        initial_units=resolved["initial_units"],
        rules=resolved["rules"],
        steps=resolved["steps"],
        arity=resolved["arity"],
        distractors=resolved["distractors"],
        count=args.count,
        seed=args.seed,
        pair=pair,  # type: ignore[arg-type]
        budget=args.budget,
        klass=getattr(args, "class"),
        max_attempts=args.max_attempts or 0,
        max_states=args.max_states,
    )
    generate.validate_config(cfg)
    cfg = generate.apply_family(cfg)
    generate.validate_config(cfg)

    out_dir = args.out
    worlds_path = os.path.join(out_dir, "worlds.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    fingerprint_path = os.path.join(out_dir, "fingerprints.json")

    accepted: dict[int, World] = {}
    fingerprints: dict[str, list[int]] = defaultdict(list)
    worlds_by_slot: dict[int, World] = {}
    rejections: Counter = Counter()
    attempted = 0

    if os.path.exists(out_dir) and os.listdir(out_dir):
        if not args.resume:
            raise SystemExit(f"输出目录已存在且非空: {out_dir}；如需续跑请加 --resume")
        if not os.path.exists(manifest_path):
            raise SystemExit(f"--resume 需要已有 manifest.json: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            old = json.load(handle)
        if old.get("config") != cfg.to_dict():
            raise SystemExit("--resume 的配置与原 manifest 不一致，拒绝续跑")
        for payload in env.iter_jsonl(worlds_path, allow_missing=True):
            world = World.from_dict(payload)
            slot = int(world.meta.get("candidate_index", -1))
            worlds_by_slot[slot] = world
            accepted[slot] = world
            fingerprints[env.structure_fingerprint(world)].append(slot)
        attempted = int(old.get("audit", {}).get("attempted", 0))
        rejections.update(old.get("audit", {}).get("rejections", {}))
    else:
        os.makedirs(out_dir, exist_ok=True)

    slots = generate.structure_slots(cfg)
    todo = [i for i in range(cfg.count) if i not in accepted]
    per_slot = max(8, cfg.effective_max_attempts // max(1, cfg.count))
    progress = Progress(cfg.count, "generate", done=len(accepted))
    iso_unresolved = 0

    rounds = 0
    while todo and attempted < cfg.effective_max_attempts and rounds < 4:
        lo = rounds * per_slot
        hi = lo + per_slot
        jobs = [(cfg, slots[i], i, lo, hi) for i in todo]
        results: list[tuple] = []
        if args.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for item in pool.map(generate.slot_worker, jobs, chunksize=4):
                    results.append(item)
        else:
            results = [generate.slot_worker(job) for job in jobs]

        next_todo: list[int] = []
        for slot, candidate, used, slot_rejections in results:
            attempted += used
            rejections.update(slot_rejections)
            if candidate is None:
                next_todo.append(slot)
                continue
            dup = False
            for other_slot in fingerprints.get(candidate.fingerprint, []):
                verdict, _ = env.isomorphic(candidate.world, accepted[other_slot])
                if verdict is None:
                    iso_unresolved += 1
                    continue
                if verdict:
                    dup = True
                    break
            if dup:
                rejections["structure_duplicate"] += 1
                next_todo.append(slot)
                continue
            accepted[slot] = candidate.world
            fingerprints[candidate.fingerprint].append(slot)
            progress.advance(1)
        todo = next_todo
        rounds += 1
    progress.close()

    ordered = [accepted[i] for i in sorted(accepted)]
    tmp = worlds_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for world in ordered:
            handle.write(env.dump_jsonl_line(world.to_dict()))
    os.replace(tmp, worlds_path)

    structure_counts = Counter(str(w.meta.get("structure")) for w in ordered)
    quota = (
        generate.allocate_quota(cfg.count)
        if cfg.structure == "mixed" and cfg.family not in generate.TRAP_FAMILIES
        else {cfg.structure if cfg.structure != "mixed" else "trap": cfg.count}
    )
    audit = {
        "requested": cfg.count,
        "accepted": len(ordered),
        "attempted": attempted,
        "rejections": dict(rejections.most_common()),
        "structure_quota": quota,
        "structure_counts": dict(structure_counts),
        "distinct_groups": len({w.group_id for w in ordered}),
        "distinct_fingerprints": len(fingerprints),
        "iso_check_unresolved": iso_unresolved,
        "witness_steps": quantiles([w.meta.get("witness_steps", 0) for w in ordered]),
        "rule_count": quantiles([len(w.rules) for w in ordered]),
        "resource_count": quantiles([w.n_resources for w in ordered]),
        "initial_units": quantiles([w.total_units for w in ordered]),
    }
    if cfg.pair:
        audit["qualification_class"] = dict(
            Counter(
                (w.meta.get("qualification") or {}).get("class", "none") for w in ordered
            )
        )
    write_json(
        manifest_path,
        {
            "schema_version": env.SCHEMA_VERSION,
            "command": "generate",
            "config": cfg.to_dict(),
            "audit": audit,
            "outputs": {"worlds": os.path.basename(worlds_path), "sha256": file_sha256(worlds_path)},
        },
    )
    write_json(
        fingerprint_path,
        {"fingerprints": {fp: len(slots_) for fp, slots_ in sorted(fingerprints.items())}},
    )

    print(
        f"generate: accepted={len(ordered)}/{cfg.count} attempted={attempted} "
        f"groups={audit['distinct_groups']}",
        file=sys.stderr,
    )
    if len(ordered) < cfg.count:
        print(
            f"候选不足：缺 {cfg.count - len(ordered)} 条，已保存现有结果并记入 manifest",
            file=sys.stderr,
        )
        return 3
    return 0


# --------------------------------------------------------------------------- #
# solve
# --------------------------------------------------------------------------- #


def _solve_job(args: tuple) -> dict:
    (
        payload,
        mode,
        max_expansions,
        max_states,
        criticality,
        criticality_states,
        max_trace_lines,
        vector_format,
    ) = args
    world = World.from_dict(payload)
    limits = search.Limits(
        max_expansions=max_expansions,
        max_states=max_states,
        max_trace_lines=max_trace_lines,
        vector_format=vector_format,
    )
    result = search.solve(world, mode, limits)
    if criticality and result.solved and result.plan is not None:
        result.criticality = env.plan_outcome(world, result.plan, criticality_states)
    return result.to_dict()


def cmd_solve(args: argparse.Namespace) -> int:
    worlds = env.load_worlds(args.worlds)
    modes = search.parse_modes(args.modes)
    out_path = args.out
    stats_path = _stem(out_path) + ".stats.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    done: set[tuple[str, str]] = set()
    existing: list[str] = []
    if args.resume and os.path.exists(out_path):
        for payload in env.iter_jsonl(out_path):
            key = (str(payload["world_id"]), str(payload["mode"]))
            if key in done:
                continue
            done.add(key)
            existing.append(env.dump_jsonl_line(payload))

    jobs = [
        (
            worlds[wid].to_dict(),
            mode,
            args.max_expansions,
            args.max_states,
            args.criticality,
            args.criticality_states,
            args.max_trace_lines,
            args.vector_format,
        )
        for wid in worlds
        for mode in modes
        if (wid, mode) not in done
    ]
    total = len(worlds) * len(modes)
    progress = Progress(total, "solve", done=len(done))

    records: list[dict] = []
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for line in existing:
            handle.write(line)
        if args.workers > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                stream: Iterable[dict] = pool.map(_solve_job, jobs, chunksize=8)
                for payload in stream:
                    handle.write(env.dump_jsonl_line(payload))
                    records.append(payload)
                    progress.advance(1)
        else:
            for job in jobs:
                payload = _solve_job(job)
                handle.write(env.dump_jsonl_line(payload))
                records.append(payload)
                progress.advance(1)
    os.replace(tmp, out_path)
    progress.close()

    all_records = [json.loads(line) for line in existing] + records
    divergence_path = None
    if len(modes) >= 2:
        divergence_path = _stem(out_path) + ".divergence.jsonl"
        _write_divergence(divergence_path, worlds, modes, all_records)
    stats = _solve_stats(worlds, modes, all_records)
    write_json(
        stats_path,
        {
            "schema_version": env.SCHEMA_VERSION,
            "command": "solve",
            "config": {
                "worlds": args.worlds,
                "worlds_sha256": file_sha256(args.worlds),
                "modes": list(modes),
                "max_expansions": args.max_expansions,
                "max_states": args.max_states,
                "max_trace_lines": args.max_trace_lines,
                "vector_format": args.vector_format,
                "solver_version": env.SOLVER_VERSION,
                "criticality": args.criticality,
                "criticality_states": args.criticality_states,
                "divergence": os.path.basename(divergence_path) if divergence_path else None,
            },
            "audit": stats,
        },
    )
    n_error = stats["status_counts"].get(search.STATUS_ERROR, 0)
    print(
        f"solve: records={len(all_records)} solved="
        f"{stats['status_counts'].get(search.STATUS_SOLVED, 0)} error={n_error}",
        file=sys.stderr,
    )
    if n_error:
        print("存在 error 状态记录，属于实现异常，请检查", file=sys.stderr)
        return 4
    return 0


def _write_divergence(
    path: str, worlds: dict[str, World], modes: Sequence[str], records: Sequence[dict]
) -> None:
    """逐题逐模式对写出第 9.4 节的分歧结构量。"""
    by_world: dict[str, dict[str, SolveResult]] = defaultdict(dict)
    for payload in records:
        by_world[str(payload["world_id"])][str(payload["mode"])] = SolveResult.from_dict(payload)
    pairs = search.mode_pairs(modes)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for wid in worlds:
            per_mode = by_world.get(wid, {})
            for pair in pairs:
                left, right = per_mode.get(pair[0]), per_mode.get(pair[1])
                if left is None or right is None:
                    continue
                handle.write(
                    env.dump_jsonl_line(search.divergence_record(wid, pair, left, right))
                )
    os.replace(tmp, path)


def _solve_stats(worlds: dict[str, World], modes: Sequence[str], records: Sequence[dict]) -> dict:
    by_world: dict[str, dict[str, dict]] = defaultdict(dict)
    status_counts: Counter = Counter()
    per_structure_mode: dict[str, Counter] = defaultdict(Counter)
    plan_len: dict[str, list[int]] = defaultdict(list)
    checked: dict[str, list[int]] = defaultdict(list)
    frontier: dict[str, list[int]] = defaultdict(list)
    trace_chars: dict[str, list[int]] = defaultdict(list)

    for payload in records:
        wid = str(payload["world_id"])
        mode = str(payload["mode"])
        by_world[wid][mode] = payload
        status_counts[payload["status"]] += 1
        structure = str(worlds[wid].meta.get("structure", "")) if wid in worlds else ""
        per_structure_mode[structure][f"{mode}:{payload['status']}"] += 1
        if payload["status"] == search.STATUS_SOLVED:
            plan_len[mode].append(len(payload["plan"] or []))
            trace_chars[mode].append(sum(len(line) + 1 for line in payload["trace"]))
        checked[mode].append(int(payload["cost"]["checked_nodes"]))
        frontier[mode].append(int(payload["cost"]["max_frontier"]))

    pair_stats = []
    if len(modes) >= 2:
        for i, left in enumerate(modes):
            for right in modes[i + 1 :]:
                same_plan = same_trace = comparable = 0
                first_diff: list[int] = []
                for wid, per_mode in by_world.items():
                    a, b = per_mode.get(left), per_mode.get(right)
                    if not a or not b:
                        continue
                    if a["status"] != search.STATUS_SOLVED or b["status"] != search.STATUS_SOLVED:
                        continue
                    comparable += 1
                    if a["plan"] == b["plan"]:
                        same_plan += 1
                    if a["trace"] == b["trace"]:
                        same_trace += 1
                    else:
                        first_diff.append(_first_difference(a["trace"], b["trace"]))
                pair_stats.append(
                    {
                        "pair": [left, right],
                        "comparable": comparable,
                        "same_plan": same_plan,
                        "same_trace": same_trace,
                        "first_different_event": quantiles(first_diff),
                    }
                )

    identifiability: Counter = Counter()
    for wid, per_mode in by_world.items():
        solved_modes = [m for m, p in per_mode.items() if p["status"] == search.STATUS_SOLVED]
        distinct_traces = len({tuple(per_mode[m]["trace"]) for m in solved_modes})
        identifiability[f"solved={len(solved_modes)},distinct_traces={distinct_traces}"] += 1

    step_kinds: Counter = Counter()
    for payload in records:
        crit = payload.get("criticality")
        if not crit:
            continue
        for key in (
            "forced_steps",
            "free_steps",
            "wasteful_steps",
            "critical_steps",
            "doomed_steps",
            "unknown_steps",
        ):
            step_kinds[key] += int(crit.get(key, 0))

    return {
        "worlds": len(worlds),
        "modes": list(modes),
        "status_counts": dict(status_counts),
        "step_kinds": dict(step_kinds) or None,
        "per_structure_mode": {k: dict(v) for k, v in sorted(per_structure_mode.items())},
        "plan_length": {m: quantiles(v) for m, v in plan_len.items()},
        "checked_nodes": {m: quantiles(v) for m, v in checked.items()},
        "max_frontier": {m: quantiles(v) for m, v in frontier.items()},
        "trace_chars": {m: quantiles(v) for m, v in trace_chars.items()},
        "mode_pairs": pair_stats,
        "identifiability": dict(identifiability),
    }


def _first_difference(left: Sequence[str], right: Sequence[str]) -> int:
    for i, (a, b) in enumerate(zip(left, right), start=1):
        if a != b:
            return i
    return min(len(left), len(right)) + 1


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def cmd_export(args: argparse.Namespace) -> int:
    qa.check_protocol(args.view, args.thinking)
    modes = search.parse_modes(args.modes)
    if args.selection == "uniform" and not args.require_all_modes:
        raise SystemExit("首版规定 --selection uniform 必须同时启用 --require-all-modes")
    if args.selection == "budget-aware" and args.budget is None:
        raise SystemExit("--selection budget-aware 必须提供 --budget")

    worlds = env.load_worlds(args.worlds)
    solutions: dict[str, dict[str, SolveResult]] = defaultdict(dict)
    for payload in env.iter_jsonl(args.solutions):
        result = SolveResult.from_dict(payload)
        if result.world_id not in worlds:
            raise SystemExit(f"solutions 中的 world_id 不在 worlds 内: {result.world_id}")
        solutions[result.world_id][result.mode] = result

    if args.budget is not None:
        for wid, per_mode in solutions.items():
            for mode, result in per_mode.items():
                if result.limits.max_expansions < args.budget:
                    raise SystemExit(
                        f"--budget {args.budget} 超过 {wid}:{mode} 的 max_expansions "
                        f"{result.limits.max_expansions}，无法判定预算内结果"
                    )

    requested_ids: list[str] | None = None
    if args.eligible_worlds:
        with open(args.eligible_worlds, encoding="utf-8") as handle:
            requested_ids = [ln.strip() for ln in handle if ln.strip()]
        unknown = [wid for wid in requested_ids if wid not in worlds]
        if unknown:
            raise SystemExit(f"--eligible-worlds 含未知 world_id：{unknown[:5]}")

    pool = requested_ids if requested_ids is not None else list(worlds)
    directions = search.modes_direction_set(modes)
    rng = random.Random(args.seed)
    rejections: Counter = Counter()

    tokenizer = qa.load_tokenizer(args.tokenizer) if args.tokenizer else None
    if args.max_tokens is not None and tokenizer is None:
        raise SystemExit("--max-tokens 必须与 --tokenizer 同时提供")

    per_world: dict[str, list[qa.QARecord]] = {}
    for wid in pool:
        world = worlds[wid]
        per_mode = solutions.get(wid, {})
        available = [m for m in modes if m in per_mode and per_mode[m].solved]
        if args.require_all_modes and len(available) != len(modes):
            rejections["missing_mode"] += 1
            continue
        if not available:
            rejections["no_solved_mode"] += 1
            continue

        chosen = available
        if args.selection == "uniform":
            chosen = [rng.choice(available)]
        elif args.selection == "budget-aware":
            verdicts = {m: search.cost_within_budget(per_mode[m], args.budget) for m in modes if m in per_mode}
            if any(v is None for v in verdicts.values()) or len(verdicts) != len(modes):
                rejections["budget_unknown"] += 1
                continue
            feasible = [m for m in modes if verdicts.get(m)]
            if not feasible:
                rejections["budget_infeasible"] += 1
                continue
            chosen = [rng.choice(feasible)]

        records: list[qa.QARecord] = []
        failed = False
        for mode in chosen:
            result = per_mode[mode]
            check = verify.verify_search(world, result.trace, modes=(mode,), budget=args.budget)
            if not (check.plan_valid and check.search_valid and check.plan_trace_consistent):
                rejections["reverify_failed"] += 1
                failed = True
                break
            record = qa.render_qa(
                world,
                result,
                view=args.view,
                thinking=args.thinking,
                directions=directions,
                budget=args.budget,
                prompt_mode=args.prompt_mode,
                prompt_style=args.prompt_style,
            )
            if tokenizer is not None:
                record.token_lengths = qa.measure_tokens(tokenizer, record)
                if args.max_tokens is not None and record.token_lengths["total"] > args.max_tokens:
                    rejections["too_long"] += 1
                    failed = True
                    break
            records.append(record)
        if failed or not records:
            continue
        per_world[wid] = records

    if args.limit_worlds is not None and len(per_world) > args.limit_worlds:
        keep = _stratified_sample(per_world, worlds, args.limit_worlds, args.seed)
        per_world = {wid: per_world[wid] for wid in keep}

    # 目标权重：同一 Q 下对实际可用来源模式等权分配。
    out_records: list[qa.QARecord] = []
    for wid in sorted(per_world):
        groups: dict[str, list[qa.QARecord]] = defaultdict(list)
        for record in per_world[wid]:
            groups[record.question].append(record)
        for bucket in groups.values():
            weight = 1.0 / len(bucket)
            for record in bucket:
                record.target_weight = weight
                out_records.append(record)

    out_path = args.out
    stem = _stem(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for record in out_records:
            handle.write(env.dump_jsonl_line(record.to_dict()))
    os.replace(tmp, out_path)

    kept_ids = sorted(per_world)
    tmp_ids = stem + ".world_ids.txt.tmp"
    with open(tmp_ids, "w", encoding="utf-8") as handle:
        for wid in kept_ids:
            handle.write(wid + "\n")
    os.replace(tmp_ids, stem + ".world_ids.txt")

    audit = _export_stats(worlds, out_records, rejections, pool)
    write_json(
        stem + ".manifest.json",
        {
            "schema_version": env.SCHEMA_VERSION,
            "command": "export",
            "config": {
                "worlds": args.worlds,
                "worlds_sha256": file_sha256(args.worlds),
                "solutions": args.solutions,
                "solutions_sha256": file_sha256(args.solutions),
                "modes": list(modes),
                "view": args.view,
                "thinking": args.thinking,
                "selection": args.selection,
                "budget": args.budget,
                "require_all_modes": args.require_all_modes,
                "limit_worlds": args.limit_worlds,
                "eligible_worlds": args.eligible_worlds,
                "seed": args.seed,
                "prompt_mode": args.prompt_mode,
                "prompt_style": args.prompt_style,
                "prompt_variant": qa.prompt_variant(
                    args.view, directions, _vector_format_of(solutions), args.prompt_style
                ),
                "tokenizer": args.tokenizer,
                "max_tokens": args.max_tokens,
                "exporter_version": env.EXPORTER_VERSION,
            },
            "audit": audit,
        },
    )

    print(
        f"export: worlds={len(kept_ids)} qa={len(out_records)} "
        f"distinct_targets={audit['distinct_target_hash']}",
        file=sys.stderr,
    )
    if requested_ids is not None:
        missing = [wid for wid in requested_ids if wid not in per_world]
        if missing:
            print(
                f"请求的 {len(missing)} 个 world_id 被拒绝，配对不完整：{missing[:5]}",
                file=sys.stderr,
            )
            return 5
    return 0


def _vector_format_of(solutions: dict) -> str:
    """从 solutions 读出统一的向量编码；混用时报错。"""
    formats = {
        r.limits.vector_format for per_mode in solutions.values() for r in per_mode.values()
    }
    if len(formats) > 1:
        raise SystemExit(f"solutions 混用了多种向量编码: {sorted(formats)}")
    return next(iter(formats)) if formats else env.VECTOR_DENSE


def _stratified_sample(
    per_world: dict[str, list[qa.QARecord]],
    worlds: dict[str, World],
    limit: int,
    seed: int,
) -> list[str]:
    """按主结构分层的稳定 seeded 顺序选取，保持输入结构比例。"""
    strata: dict[str, list[str]] = defaultdict(list)
    for wid in sorted(per_world):
        strata[str(worlds[wid].meta.get("structure", ""))].append(wid)
    total = sum(len(v) for v in strata.values())
    raw = {k: limit * len(v) / total for k, v in strata.items()}
    take = {k: int(v) for k, v in raw.items()}
    left = limit - sum(take.values())
    order = sorted(strata, key=lambda k: (-(raw[k] - take[k]), k))
    for k in order[:left]:
        take[k] += 1
    keep: list[str] = []
    for key in sorted(strata):
        pool = list(strata[key])
        random.Random(f"{seed}:{key}").shuffle(pool)
        keep.extend(pool[: take[key]])
    return sorted(keep)


def _export_stats(
    worlds: dict[str, World],
    records: Sequence[qa.QARecord],
    rejections: Counter,
    pool: Sequence[str],
) -> dict:
    by_structure_in: Counter = Counter(str(worlds[w].meta.get("structure", "")) for w in pool)
    kept_worlds = {r.world_id for r in records}
    by_structure_out: Counter = Counter(
        str(worlds[w].meta.get("structure", "")) for w in kept_worlds
    )
    weights: dict[str, float] = defaultdict(float)
    for record in records:
        weights[record.target_hash] += record.target_weight
    return {
        "input_worlds": len(pool),
        "output_worlds": len(kept_worlds),
        "qa_records": len(records),
        "distinct_target_hash": len(weights),
        "distinct_questions": len({r.question for r in records}),
        "distinct_assistants": len({r.assistant for r in records}),
        "source_mode_counts": dict(Counter(r.source_mode for r in records)),
        "structure_retention": {
            key: {
                "input": by_structure_in.get(key, 0),
                "output": by_structure_out.get(key, 0),
                "rate": round(by_structure_out.get(key, 0) / by_structure_in[key], 4)
                if by_structure_in.get(key)
                else None,
            }
            for key in sorted(by_structure_in)
        },
        "weight_sum": round(sum(weights.values()), 6),
        "plan_length": quantiles([r.plan_length for r in records]),
        "trace_events": quantiles([r.trace_events for r in records]),
        "question_chars": quantiles([r.char_lengths["question"] for r in records]),
        "assistant_chars": quantiles([r.char_lengths["assistant"] for r in records]),
        "token_total": quantiles(
            [r.token_lengths["total"] for r in records if r.token_lengths]
        ),
        "rejections": dict(rejections.most_common()),
    }


def _stem(path: str) -> str:
    return path[:-6] if path.endswith(".jsonl") else path


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


def _verify_records(args: argparse.Namespace, worlds: dict, *, unique_ids: bool = False) -> list:
    modes = search.parse_modes(args.modes)
    shared_lines: dict[str, int] = {}
    if args.divergence:
        target = list(modes[:2])
        if len(target) < 2:
            raise SystemExit("--divergence 需要 --modes 至少给出两个模式")
        for payload in env.iter_jsonl(args.divergence):
            if list(payload.get("pair") or []) != target:
                continue
            if payload.get("shared_lines") is None:
                continue
            shared_lines[str(payload["world_id"])] = int(payload["shared_lines"])
        if not shared_lines:
            raise SystemExit(
                f"--divergence 中没有模式对 {target[0]}/{target[1]} 的可用记录"
            )
    records = []
    seen = set()
    for index, payload in enumerate(env.iter_jsonl(args.input), start=1):
        record_id, world_id, text = verify.coerce_prediction(payload, index)
        if unique_ids and record_id in seen:
            raise env.WorldError(f"record_id 重复：{record_id}；同题多次回答需使用不同 record_id")
        seen.add(record_id)
        if world_id not in worlds:
            raise SystemExit(f"第 {index} 条记录的 world_id 不存在: {world_id}")
        result = verify.verify_record(
            worlds[world_id],
            text,
            kind=args.kind,
            record_id=record_id,
            modes=modes,
            budget=args.budget,
            prefill=args.prefill,
            shared_lines=shared_lines.get(world_id),
            criticality=args.criticality,
            criticality_states=args.criticality_states,
            max_plan_steps=args.max_plan_steps or None,
            max_events=args.max_events or None,
            protocol=getattr(args,'protocol','standard'),
        )
        records.append((payload, result))

    return records


def cmd_verify(args: argparse.Namespace) -> int:
    if getattr(args, "tag", None) is not None:
        return _verify_with_tag(args)
    if getattr(args, "reference", None):
        raise env.WorldError("--reference 仅用于 --tag 的 QA 导出")
    worlds = env.load_worlds(args.worlds)
    modes = search.parse_modes(args.modes)
    results = [result for _, result in _verify_records(args, worlds)]

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(env.dump_jsonl_line(result.to_dict(include_text=False)))
        os.replace(tmp, args.out)
        write_json(
            _stem(args.out) + ".stats.json",
            {
                "schema_version": env.SCHEMA_VERSION,
                "command": "verify",
                "config": {
                    "worlds": args.worlds,
                    "input": args.input,
                    "kind": args.kind,
                    "modes": list(modes),
                    "budget": args.budget,
                    "prefill": args.prefill,
                    "divergence": args.divergence,
                    "criticality": args.criticality,
                    "criticality_states": args.criticality_states,
                    "max_plan_steps": args.max_plan_steps,
                    "max_events": args.max_events,
                    "verification_policy": args.protocol if getattr(args,'protocol','standard')!='standard' else verify.VERIFICATION_POLICY,
                },
                "audit": _verify_stats(results),
            },
        )

    audit = _verify_stats(results)
    print(
        f"verify: records={audit['records']} plan_valid={audit['plan_valid']} "
        f"search_valid={audit['search_valid']} passed={audit['passed']}",
        file=sys.stderr,
    )
    if args.strict and audit["passed"] != audit["records"]:
        return 6
    return 0


def _rate(values: Sequence[bool | None]) -> dict:
    passed = sum(v is True for v in values)
    failed = sum(v is False for v in values)
    measured = passed + failed
    return {"passed": passed, "failed": failed, "unmeasured": len(values)-measured,
            "denominator": measured, "rate": passed/measured if measured else None}


def _tagged_verify_summary(rows: Sequence[dict], world_ids: set[str], audit: dict) -> dict:
    received = {r["world_id"] for r in rows}
    attempts = Counter(r["world_id"] for r in rows)
    metrics = {name: _rate([r[name] for r in rows]) for name in
               ("plan_valid", "search_valid", "plan_trace_consistent", "within_budget", "hack")}
    metrics["syntax_complete"] = _rate([r["parse_status"] == verify.PARSE_COMPLETE for r in rows])
    metrics["mode_compatible"] = _rate([
        bool(r["compatible_modes"]) if r["compatible_modes"] is not None else None for r in rows])
    metrics["passed"] = {"passed": audit["passed"], "failed": len(rows)-audit["passed"],
                         "unmeasured": 0, "denominator": len(rows),
                         "rate": audit["passed"]/len(rows) if rows else None}
    return {"records": len(rows), "coverage": {"worlds_in_input": len(world_ids),
        "worlds_with_predictions": len(received), "worlds_without_predictions": len(world_ids-received),
        "missing_world_ids": sorted(world_ids-received),
        "attempts_per_world": dict(Counter(str(n) for n in attempts.values()))},
        "metrics": metrics, "audit": audit,
        "denominator_policy": "Rates use measured records; null is unmeasured. Passed uses all received records. Coverage is reported separately.",
        "mode_policy": "Mode compatibility is reported separately from generic search/answer validity; passed does not require a fixed Teacher mode or staying within a supplied budget."}


def _write_verify_markdown(path: Path, tag: str, summary: dict) -> None:
    """Keep the reading summary focused; full diagnostics remain in summary.json."""
    def percent(count: int, total: int) -> str:
        return f"{count / total:.2%}" if total else "—"

    def score(count: int, total: int) -> str:
        return f"{percent(count, total)}（{count}/{total}）"

    kind = summary.get("kind")
    is_search = kind == "search"
    metrics = summary["metrics"]
    n = summary["records"]
    primary = summary.get("primary_metric", {}) if is_search else {}
    passed = primary.get("count", metrics["passed"]["passed"])
    label = {"search": "联合成功率", "plan": "方案通过率", "prefix": "前缀通过率"}.get(kind, "验证通过率")
    c = summary["coverage"]
    groups = summary.get("all_records_metrics", {})
    lines = [f"# 验证结果：{tag}", "",
             f"**{label}：{score(passed, n)}**", ""]
    if is_search:
        lines.extend(["成功条件：最终方案正确且搜索过程合法；分母为全部收到的回答。", ""])
    coverage = (f"问题覆盖 {c['worlds_with_predictions']}/{c['worlds_in_input']}"
                f" · 回答 {n} 条")
    if c["worlds_without_predictions"]:
        coverage += f" · **缺少 {c['worlds_without_predictions']} 题回答，当前结果为部分覆盖**"
    lines.append(coverage)
    cut = groups.get("all", {}).get("truncation", {})
    if cut.get("denominator"):
        truncation = f"截断率：{score(cut['count'], cut['denominator'])}"
        if cut.get("unmeasured"):
            truncation += f"；另有 {cut['unmeasured']} 条缺少结束原因"
    else:
        truncation = "截断率：未测量（缺少结束原因）"
    lines.extend(["", truncation, ""])

    if is_search:
        hack = metrics["hack"]["passed"]
        wrong = metrics["plan_valid"]["failed"]
        outcomes = [("成功：方案正确、过程合法", passed),
                    ("失败：方案正确、过程无效（hack）", hack),
                    ("失败：方案错误", wrong)]
        unknown = n - passed - hack - wrong
        if unknown:
            outcomes.append(("未判定（计入主指标分母）", unknown))
        lines.extend(["## 回答分类", "",
                      "| 结果 | 回答数 | 占全部回答 |", "|---|---:|---:|"])
        lines.extend(f"| {name} | {count} | {percent(count, n)} |" for name, count in outcomes)
        lines.extend(["", f"最终方案正确率（辅助）：{score(metrics['plan_valid']['passed'], n)}，包含成功与 hack。", ""])

    # Prefix verification has no joint/answer success metric per tier.
    tier_groups = [(key.removeprefix("tier:"), row) for key, row in groups.items() if key.startswith("tier:")]
    if tier_groups and kind in ("search", "plan"):
        lines.extend(["## 难度分层", ""])
        metric_key = "joint_correct" if is_search else "answer_correct"
        lines.extend([f"| 难度 | 回答数 | {label} |" + (" Hack 占比 |" if is_search else ""),
                      "|---|---:|---:|" + ("---:|" if is_search else "")])
        for tier, row in tier_groups:
            tier_label = {"easy": "简单", "mid": "中等", "hard": "困难"}.get(tier, tier)
            # Metadata can contain Markdown separators or newlines.
            tier_label = tier_label.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            line = f"| {tier_label} | {row['n']} | {score(row['counts'][metric_key], row['n'])} |"
            if is_search:
                line += f" {score(row['counts']['hack'], row['n'])} |"
            lines.append(line)
        lines.append("")

    errors = sorted(summary["audit"]["first_error"].items(), key=lambda item: (-item[1], item[0]))
    if errors:
        lines.extend(["## 主要首错", "", "每条回答仅计首个错误，按数量列出前三类。", "",
                      "| 错误类型 | 回答数 | 占全部回答 |", "|---|---:|---:|"])
        for code, count in errors[:3]:
            name = env.ERROR_CODE_LABELS.get(code, code)
            lines.append(f"| {name}（`{code}`） | {count} | {percent(count, n)} |")
        if len(errors) > 3:
            remaining = sum(count for _, count in errors[3:])
            lines.append(f"| 其他 {len(errors) - 3} 类 | {remaining} | {percent(remaining, n)} |")
        lines.append("")
    lines.extend(["[逐条问答与判定](qa.jsonl) · [完整指标与错误分布](summary.json)", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _question_from_record(payload: dict) -> str | None:
    if "question" in payload:
        if not isinstance(payload["question"], str):
            raise env.WorldError("question 必须是字符串")
        return payload["question"]
    for message in reversed(payload.get("messages", [])):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return None


def _load_verify_references(path: str | None) -> dict:
    references = {}
    if not path:
        return references
    for index, payload in enumerate(env.iter_jsonl(path), 1):
        if "reference_answer" in payload:
            _, world_id, text = verify.coerce_prediction(
                {"world_id": payload.get("world_id"), "text": payload["reference_answer"]}, index)
        else:
            _, world_id, text = verify.coerce_prediction(payload, index)
        item = {"question": _question_from_record(payload), "reference_answer": text}
        if world_id in references and references[world_id] != item:
            raise env.WorldError(f"参考文件包含同一 world_id 的不同轨迹：{world_id}；请指定单模式参考")
        references[world_id] = item
    return references


def _verified_qa_row(payload: dict, result: verify.VerifyResult, world: World,
                     reference: dict | None = None, prefill: str | None = None) -> dict:
    reference = reference or {}
    question = _question_from_record(payload)
    if question is None:
        question = reference.get("question")
    if question is None:
        question = qa.render_header(world)
    # Retain generation metadata, but store each answer only once.
    redundant = {"text", "trace", "messages", "generated_suffix", "input_ids", "prompt",
                 "question", "model_answer", "reference_answer", "verification"}
    row = {key: value for key, value in payload.items() if key not in redundant}
    row.update(record_id=result.record_id, world_id=result.world_id, question=question,
               model_answer=verify.restore_prefill(result.text, prefill),
               verification=result.to_dict(include_text=False))
    if prefill is not None:
        row["applied_prefill"] = prefill
    answer = payload.get("reference_answer", reference.get("reference_answer"))
    if answer is not None:
        if not isinstance(answer, str):
            raise env.WorldError("reference_answer 必须是字符串")
        row["reference_answer"] = answer
    tier = world.meta.get("dataset_tier")
    if tier is not None:
        row.setdefault("tier", tier)
    return row


def _add_qa_metrics(summary: dict, records: Sequence[dict]) -> None:
    groups = {"all": list(records)}
    for tier in sorted({r["tier"] for r in records if isinstance(r.get("tier"), str)}):
        groups[f"tier:{tier}"] = [r for r in records if r.get("tier") == tier]
    summary["all_records_metrics"] = {}
    for label, group in groups.items():
        n = len(group)
        counts = {
            "joint_correct": sum(r["verification"]["plan_valid"] is True
                                 and r["verification"]["search_valid"] is True for r in group),
            "hack": sum(r["verification"].get("hack") is True for r in group),
            "answer_correct": sum(r["verification"]["plan_valid"] is True for r in group),
            "syntax_complete": sum(r["verification"]["parse_status"] == verify.PARSE_COMPLETE for r in group),
        }
        if summary["kind"] != "search":
            counts["joint_correct"] = None
            counts["hack"] = None
        # Missing finish_reason is unknown, not evidence of a nontruncated answer.
        measured = [r for r in group if r.get("finish_reason") is not None]
        summary["all_records_metrics"][label] = {
            "n": n, "counts": counts,
            "rates": {key: count / n if n and count is not None else None for key, count in counts.items()},
            "truncation": {"count": sum(r["finish_reason"] == "length" for r in measured),
                           "denominator": len(measured), "unmeasured": n - len(measured),
                           "rate": sum(r["finish_reason"] == "length" for r in measured) / len(measured)
                                   if measured else None},
        }
    if summary["kind"] == "search":
        all_records = summary["all_records_metrics"]["all"]
        summary["primary_metric"] = {
            "name": "joint_success_rate", "label": "联合成功率",
            "count": all_records["counts"]["joint_correct"],
            "denominator": all_records["n"], "rate": all_records["rates"]["joint_correct"],
            "definition": "plan_valid=true AND search_valid=true",
        }
        summary["hack_policy"] = "答案正确但过程无效（含过程错误或不完整），标记 hack=true，主指标计为失败。"

    summary["verification_policy"] = verify.VERIFICATION_POLICY


def _verify_with_tag(args: argparse.Namespace) -> int:
    tag = args.tag
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", tag):
        raise env.WorldError("--tag 需为 1–128 位字母、数字、下划线或点，首位为字母或数字")
    if args.out:
        raise env.WorldError("--tag 与 --out 不能同时使用")
    folder = RESULT_ROOT / tag
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        folder.mkdir()
    except FileExistsError as exc:
        raise env.WorldError(f"结果目录已存在，拒绝覆盖：{folder}；请使用新的 tag") from exc
    summary = {"tag": tag, "status": "running", "schema_version": env.SCHEMA_VERSION,
               "format_version": 2, "kind": args.kind,
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "settings": {name: getattr(args, name) for name in
                            ("kind", "modes", "budget", "prefill", "strict", "criticality",
                             "criticality_states", "max_plan_steps", "max_events")},
               "sources": {}}
    write_json(str(folder / "summary.json"), summary)
    temporary = folder / "qa.jsonl.tmp"
    try:
        for key in ("worlds", "input", "reference", "divergence"):
            source = getattr(args, key, None)
            if source:
                summary["sources"][key] = {"path": str(Path(source).resolve()),
                                           "sha256": file_sha256(source)}
        worlds = env.load_worlds(args.worlds)
        references = _load_verify_references(getattr(args, "reference", None))
        records = _verify_records(args, worlds, unique_ids=True)
        if not records:
            raise env.WorldError("输入中没有回答记录，不能生成验证统计")
        if getattr(args, "reference", None):
            missing = {r.world_id for _, r in records} - references.keys()
            if missing:
                raise env.WorldError(f"参考文件缺少 {len(missing)} 个问题：{sorted(missing)[:3]}")
        joined = [_verified_qa_row(payload, r, worlds[r.world_id], references.get(r.world_id), args.prefill)
                  for payload, r in records]
        with temporary.open("w", encoding="utf-8") as handle:
            for row in joined:
                handle.write(env.dump_jsonl_line(row))
        os.replace(temporary, folder / "qa.jsonl")
        results = [r for _, r in records]
        audit = _verify_stats(results)
        summary.update(_tagged_verify_summary([r.to_dict(include_text=False) for r in results],
                                             set(worlds), audit))
        _add_qa_metrics(summary, joined)
        if getattr(args,'protocol','standard')!='standard':summary['verification_policy']=args.protocol
        code = 6 if args.strict and audit["passed"] != audit["records"] else 0
        summary.update(status="complete", exit_code=code,
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        _write_verify_markdown(folder / "summary.md", tag, summary)
        write_json(str(folder / "summary.json"), summary)
        print(f"verify: records={audit['records']} plan_valid={audit['plan_valid']} "
              f"search_valid={audit['search_valid']} passed={audit['passed']}", file=sys.stderr)
        print(f"result: {folder.resolve()}", file=sys.stderr)
        return code
    except BaseException as exc:
        temporary.unlink(missing_ok=True)
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error_type=type(exc).__name__, error=str(exc),
                       finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(str(folder / "summary.json"), summary)
        raise


def _verify_stats(results: Sequence[verify.VerifyResult]) -> dict:
    return {
        "records": len(results),
        "passed": sum(1 for r in results if r.passed),
        "parse_status": dict(Counter(r.parse_status for r in results)),
        "direction": dict(Counter(str(r.direction) for r in results)),
        "plan_valid": sum(1 for r in results if r.plan_valid),
        "search_valid": sum(1 for r in results if r.search_valid),
        "plan_trace_consistent": sum(1 for r in results if r.plan_trace_consistent),
        "within_budget": sum(1 for r in results if r.within_budget),
        "first_error": dict(
            Counter(r.first_error.code for r in results if r.first_error).most_common()
        ),
        "compatible_mode_sets": dict(
            Counter(
                ",".join(r.compatible_modes) if r.compatible_modes is not None else "null"
                for r in results
            ).most_common()
        ),
        "checked_nodes": quantiles(
            [r.checked_nodes for r in results if r.checked_nodes is not None]
        ),
        "first_error_offset_to_divergence": quantiles(
            [
                r.first_error_offset_to_divergence
                for r in results
                if r.first_error_offset_to_divergence is not None
            ]
        ),
        "first_error_frontier_size": quantiles(
            [
                r.first_error_frontier_size
                for r in results
                if r.first_error_frontier_size is not None
            ]
        ),
        "plan_steps": quantiles([r.plan_steps for r in results if r.plan_steps is not None]),
        "plan_waste": quantiles([r.plan_waste for r in results if r.plan_waste is not None]),
        "state_revisits": quantiles(
            [r.state_revisits for r in results if r.state_revisits is not None]
        ),
        "units_consumed": quantiles(
            [r.units_consumed for r in results if r.units_consumed is not None]
        ),
        "steps_after_goal": quantiles(
            [r.steps_after_goal for r in results if r.steps_after_goal is not None]
        ),
        "wasteful_plans": sum(1 for r in results if r.plan_waste),
        "plans_with_revisits": sum(1 for r in results if r.state_revisits),
        "units_not_monotone": sum(1 for r in results if r.units_monotone is False),
        "search_outcome": dict(Counter(str(r.search_outcome) for r in results).most_common()),
        "vector_format": dict(Counter(str(r.vector_format) for r in results).most_common()),
        "hack": sum(1 for r in results if r.hack),
        "hack_kind": dict(
            Counter(r.hack_kind for r in results if r.hack_kind).most_common()
        ),
        "critical_step_vs_plan_valid": _critical_step_table(results),
        "first_unrecoverable_step": quantiles(
            [
                r.first_unrecoverable_step
                for r in results
                if r.first_unrecoverable_step is not None
            ]
        ),
    }


def _critical_step_table(results: Sequence[verify.VerifyResult]) -> dict:
    """首错步是否为关键步，与最终方案是否有效的 2x2 表。"""
    table: Counter = Counter()
    for r in results:
        if r.first_error_at_critical_step is None or r.plan_valid is None:
            continue
        key = (
            "critical" if r.first_error_at_critical_step else "non_critical",
            "plan_valid" if r.plan_valid else "plan_invalid",
        )
        table[key] += 1
    return {f"{a}/{b}": n for (a, b), n in sorted(table.items())} or None


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m symplan",
        description="符号资源推理环境：问题生成、自动搜索轨迹、Qwen QA 导出与验证",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成问题")
    gen.add_argument("--out", required=True, help="输出目录，写出 worlds.jsonl 与 manifest.json")
    gen.add_argument("--family", default="general", choices=generate.FAMILIES)
    gen.add_argument("--structure", default="mixed", choices=generate.STRUCTURES)
    gen.add_argument("--level", choices=tuple(generate.LEVELS))
    gen.add_argument("--resources", help="资源类型数，正整数或区间如 16:24")
    gen.add_argument("--initial-units", help="初始资源总个数，正整数或区间")
    gen.add_argument("--rules", help="规则总数，正整数或区间")
    gen.add_argument("--steps", help="植入方案长度范围")
    gen.add_argument("--arity", type=int, help="单条规则输入资源总数上限")
    gen.add_argument("--distractors", type=float, help="剩余规则槽位中分配给干扰构造的比例")
    gen.add_argument("--count", type=int, required=True, help="通过验证和去重的问题数量")
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--workers", type=int, default=1)
    gen.add_argument("--pair", help="优势筛选的两个模式，逗号分隔")
    gen.add_argument("--budget", type=int, help="筛选用的节点预算 B")
    gen.add_argument("--class", default="any", choices=generate.CLASSES, dest="class")
    gen.add_argument("--max-attempts", type=int, help="候选尝试上限，默认 50×count")
    gen.add_argument("--max-states", type=int, default=search.DEFAULT_MAX_STATES)
    gen.add_argument("--resume", action="store_true")
    gen.set_defaults(func=cmd_generate)

    slv = sub.add_parser("solve", help="生成指定模式的轨迹")
    slv.add_argument("--worlds", required=True)
    slv.add_argument("--modes", required=True, help="逗号分隔，如 forward-bfs,forward-dfs")
    slv.add_argument("--max-expansions", type=int, default=search.DEFAULT_MAX_EXPANSIONS)
    slv.add_argument("--max-states", type=int, default=search.DEFAULT_MAX_STATES)
    slv.add_argument("--out", required=True)
    slv.add_argument("--workers", type=int, default=1)
    slv.add_argument("--resume", action="store_true")
    slv.add_argument(
        "--criticality",
        action="store_true",
        help="沿每个模式的实际方案计算步骤分类与修复代价（第 6.9 节）",
    )
    slv.add_argument("--criticality-states", type=int, default=env.DEFAULT_CRITICALITY_STATES)
    slv.add_argument(
        "--max-trace-lines",
        type=int,
        default=search.DEFAULT_MAX_TRACE_LINES,
        help="单条轨迹的事件行数上限，超出时状态记 trace_limit；0 表示不限",
    )
    slv.add_argument(
        "--vector-format",
        default=env.VECTOR_DENSE,
        choices=env.VECTOR_FORMATS,
        help="事件行中向量的编码：dense 列出全部计数，sparse 只列非零项",
    )
    slv.set_defaults(func=cmd_solve)

    exp = sub.add_parser("export", help="导出 Qwen QA")
    exp.add_argument("--worlds", required=True)
    exp.add_argument("--solutions", required=True)
    exp.add_argument("--modes", required=True)
    exp.add_argument("--view", required=True, choices=qa.VIEWS)
    exp.add_argument("--thinking", required=True, choices=qa.THINKING)
    exp.add_argument("--selection", default="all", choices=("all", "uniform", "budget-aware"))
    exp.add_argument("--budget", type=int)
    exp.add_argument("--require-all-modes", action="store_true")
    exp.add_argument("--limit-worlds", type=int)
    exp.add_argument("--eligible-worlds")
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument("--prompt-mode", default=qa.PROMPT_MODE_NONE, choices=qa.PROMPT_MODES)
    exp.add_argument("--prompt-style", default=qa.PROMPT_STRUCTURED, choices=qa.PROMPT_STYLES,
                     help="structured 仅保留题目字段；detailed 保留历史教程说明")
    exp.add_argument("--tokenizer")
    exp.add_argument("--max-tokens", type=int)
    exp.add_argument("--out", required=True)
    exp.set_defaults(func=cmd_export)

    ver = sub.add_parser("verify", help="验证方案、完整过程或前缀")
    ver.add_argument("--worlds", required=True)
    ver.add_argument("--input", required=True)
    ver.add_argument("--reference", help="可选单模式参考 QA，按 world_id 合并到 --tag 的 qa.jsonl")
    ver.add_argument("--kind", required=True, choices=verify.KINDS)
    ver.add_argument('--protocol',choices=['standard','search_recheck_v1'],default='standard')
    ver.add_argument("--modes", default=",".join(search.MODES))
    ver.add_argument("--budget", type=int)
    ver.add_argument("--prefill", choices=("nonthink", "think"))
    verify_output = ver.add_mutually_exclusive_group()
    verify_output.add_argument("--out")
    verify_output.add_argument("--tag", help="写入当前目录下的 result/<tag>/，同名目录拒绝覆盖")
    ver.add_argument("--strict", action="store_true")
    ver.add_argument("--divergence", help="solve 写出的 divergence.jsonl，用于首错位置相对分歧位置的报告")
    ver.add_argument(
        "--criticality",
        action="store_true",
        help="按模型自身前缀判定首错步是否为关键步，并给出首错 EXPAND 处的 frontier 规模",
    )
    ver.add_argument("--criticality-states", type=int, default=env.DEFAULT_CRITICALITY_STATES)
    ver.add_argument(
        "--max-plan-steps",
        type=int,
        default=verify.DEFAULT_MAX_PLAN_STEPS,
        help="方案长度上限，超出记 PLAN_TOO_LONG；设为 0 表示不限",
    )
    ver.add_argument(
        "--max-events",
        type=int,
        default=verify.DEFAULT_MAX_EVENTS,
        help="事件行数上限，超出记 TRACE_TOO_LONG；设为 0 表示不限",
    )
    ver.set_defaults(func=cmd_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (env.WorldError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

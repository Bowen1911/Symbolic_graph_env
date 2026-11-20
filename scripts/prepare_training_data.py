"""Build a shared, stratified four-Teacher corpus using the symplan core.

No model inference. Immutable completed rounds support explicit resume; final
views share questions and prompts, and subset indexes avoid duplicating traces.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from symplan import env, generate, qa, search, verify
from symplan.cli import file_sha256, quantiles, write_json

MODES = search.MODES
TRAINING_PROTOCOLS = (("search", "on"),)
TRAINING_VIEWS = tuple(view for view, _ in TRAINING_PROTOCOLS)
TIERS = {
    "easy": dict(resources=(8, 12), initial_units=(3, 5), rules=(10, 18),
                 steps=(2, 4), arity=2, distractors=0.25),
    "mid": dict(resources=(16, 24), initial_units=(4, 6), rules=(16, 26),
                steps=(4, 6), arity=2, distractors=0.4),
}
_tokenizer = None


class LocalTokenizer:
    def __init__(self, path):
        from transformers import AutoTokenizer
        # tokenizer_config adds <think>/</think> token definitions absent from
        # tokenizer.json. Loading the raw JSON alone changes token IDs/counts.
        self.backend = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False).backend_tokenizer

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.backend.encode(text, add_special_tokens=False).ids}


def init_worker(path):
    global _tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _tokenizer = LocalTokenizer(path)


def token_lengths(record):
    lengths = qa.measure_tokens(_tokenizer, record)
    exact = len(_tokenizer(qa.render_train_text(record.question, record.assistant))["input_ids"])
    lengths["segmented_total"] = lengths["total"]
    lengths["total"] = exact
    return lengths


def prepare_one(job):
    tier, structure, seed, slot, limits_dict, max_tokens = job
    cfg = generate.GenConfig(family="general", structure=structure, level=None,
                             count=1, seed=seed, **TIERS[tier])
    generate.validate_config(cfg)
    _, candidate, attempts, failures = generate.slot_worker((cfg, structure, slot, 0, 50))
    base = {"tier": tier, "structure": structure, "seed": seed, "slot": slot,
            "attempts": attempts, "generation_rejections": dict(failures)}
    if candidate is None:
        return {**base, "reason": "generation_failed"}
    world = candidate.world
    world.meta["dataset_tier"] = tier
    limits = search.Limits(**limits_dict)
    base.update(world=world.to_dict(), fingerprint=candidate.fingerprint,
                modes={mode: {"status": "not_run_after_rejection"} for mode in MODES})
    solved = {}
    rendered = {}
    # Reject as soon as a necessary condition fails. This changes diagnostics
    # coverage only; all four modes are still solved and replayed for every
    # accepted world. Keep serialization in the original MODES order below.
    for mode in ("backward-bfs", "backward-dfs", "forward-bfs", "forward-dfs"):
        result = search.solve(world, mode, limits)
        base["modes"][mode] = {"status": result.status, "cost": result.to_dict()["cost"],
                              "trace_lines": len(result.trace),
                              "plan_steps": len(result.plan) if result.plan is not None else None}
        if result.status in ("error", "exhausted"):
            raise RuntimeError(f"Unexpected solver failure for witnessed world {world.world_id}: {base['modes']}")
        if not result.solved:
            return {**base, "reason": "missing_mode"}
        solved[mode] = result
        rendered[mode] = {}
        for view, thinking in TRAINING_PROTOCOLS:
            record = qa.render_qa(world, result, view=view, thinking=thinking,
                                 directions=("forward", "backward"), prompt_style=qa.PROMPT_STRUCTURED)
            record.target_weight = 0.25
            record.token_lengths = token_lengths(record)
            base["modes"][mode][view + "_tokens"] = record.token_lengths["total"]
            if record.token_lengths["total"] > max_tokens:
                return {**base, "reason": "too_long"}
            rendered[mode][view] = record.to_dict()
    results = [solved[mode] for mode in MODES]
    records = {view: [] for view in TRAINING_VIEWS}
    for result in results:
        for view in TRAINING_VIEWS:
            records[view].append(rendered[result.mode][view])
    for result in results:
        for view in TRAINING_VIEWS:
            record = next(r for r in records[view] if r["source_mode"] == result.mode)
            check = verify.verify_record(world, record["messages"][1]["content"],
                                         kind=view, modes=(result.mode,))
            if not check.plan_valid or (view == "search" and not (
                    check.search_valid and check.plan_trace_consistent
                    and result.mode in check.compatible_modes)):
                raise RuntimeError(f"Independent replay failed: {world.world_id} {result.mode} {view}")
    return {**base, "reason": "eligible", "solutions": [r.to_dict() for r in results],
            "records": records,
            "divergence": [search.divergence_record(world.world_id, pair, results[MODES.index(pair[0])],
                            results[MODES.index(pair[1])]) for pair in search.mode_pairs(MODES)]}


def dump(handle, payload):
    handle.write(env.dump_jsonl_line(payload))


def quotas(n):
    if n % 2:
        raise ValueError("Question count must be even for the equal tier allocation")
    return {(tier, structure): count for tier in TIERS
            for structure, count in generate.allocate_quota(n // 2).items()}


def config_for(args):
    source_paths = sorted((ROOT / "symplan").glob("*.py")) + [Path(__file__)]
    return {"task_name": args.task_name, "phase": args.phase,
            "schema_version": env.SCHEMA_VERSION, "generator_version": env.GENERATOR_VERSION,
            "solver_version": env.SOLVER_VERSION, "train": args.train,
            "validation": args.validation, "test": args.test, "pilot": args.pilot,
            "seed": args.seed, "modes": list(MODES), "tiers": TIERS,
            "active_protocol": {"view": "search", "thinking": "on"},
            "prompt_style": qa.PROMPT_STRUCTURED, "prompt_revision": "structured-v2",
            "quota": generate.MIXED_QUOTA, "tokenizer": str(Path(args.tokenizer).resolve()),
            "tokenizer_sha256": file_sha256(str(Path(args.tokenizer) / "tokenizer.json")),
            "tokenizer_config_sha256": file_sha256(str(Path(args.tokenizer) / "tokenizer_config.json")),
            "max_tokens": args.max_tokens, "max_candidates": args.max_candidates,
            "batch_per_stratum": args.batch_per_stratum,
            "libraries": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")},
            "limits": {"max_expansions": 4096, "max_states": 100000,
                       "max_trace_lines": args.max_trace_lines, "vector_format": "sparse"},
            "source_sha256": {str(p.relative_to(ROOT)): file_sha256(str(p)) for p in source_paths}}


def save_prompts(out):
    world = env.World(symbols=("A", "B", "C", "D", "G"), initial=(1, 1, 1, 0, 0),
        goal=(0, 0, 0, 0, 1), rules=(env.Rule("r0", (1, 1, 0, 0, 0), (0, 0, 0, 1, 0)),
        env.Rule("r1", (0, 0, 1, 1, 0), (0, 0, 0, 0, 1))))
    for view in TRAINING_VIEWS:
        text = qa.render_question(world, view=view, directions=("forward", "backward"),
                                  vector_format="sparse", prompt_style=qa.PROMPT_STRUCTURED)
        (out / f"prompt_{view}.structured-v2.txt").write_text(text + "\n", encoding="utf-8")


def build(args):
    out = Path(args.out).resolve()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get("generation_intermediates_retained") is False:
            raise SystemExit("成品目录的生成中间文件已清理，不能在此目录重新生成或 --resume；请使用新的 --out 目录")
    config = json.loads(json.dumps(config_for(args)))
    cp = out / "config.json"
    if cp.exists():
        if not args.resume:
            raise SystemExit("Output exists; use explicit --resume")
        if json.loads(cp.read_text()) != config:
            raise SystemExit("Resume config/source/tokenizer mismatch")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(str(cp), config)
        save_prompts(out)
    target = quotas(args.pilot) if args.phase == "pilot" else dict(
        Counter(quotas(args.train)) + Counter(quotas(args.validation)) + Counter(quotas(args.test)))
    accepted = Counter()
    scheduled = Counter()
    seen = defaultdict(list)
    group_ids = set()
    reasons = Counter()
    generated = 0
    rounds = sorted((out / "rounds").glob("*/complete.json"))
    for marker in rounds:
        status = json.loads(marker.read_text())
        for name, digest in status["hashes"].items():
            if file_sha256(str(marker.parent / name)) != digest:
                raise SystemExit(f"Completed round corrupted: {marker.parent / name}")
        generated += status["candidates"]
        reasons.update(status["reasons"])
        for row in env.iter_jsonl(str(marker.parent / "candidates.jsonl")):
            scheduled[(row["tier"], row["structure"])] += 1
        for p in env.iter_jsonl(str(marker.parent / "worlds.jsonl")):
            world = env.World.from_dict(p)
            key = (world.meta["dataset_tier"], world.meta["structure"])
            accepted[key] += 1
            seen[env.structure_fingerprint(world)].append(world)
            group_ids.add(world.group_id)
    start = time.monotonic()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        round_no = len(rounds)
        while generated < (args.pilot if args.phase == "pilot" else args.max_candidates):
            if args.phase != "pilot" and all(accepted[k] >= n for k, n in target.items()):
                break
            jobs = []
            remaining = (args.pilot if args.phase == "pilot" else args.max_candidates) - generated
            for k, n in target.items():
                need = n - scheduled[k] if args.phase == "pilot" else n - accepted[k]
                proposed = need if args.phase == "pilot" else (args.batch_per_stratum if need > 0 else 0)
                count = min(proposed, args.batch_per_stratum, remaining - len(jobs))
                for slot in range(max(0, count)):
                    jobs.append((*k, args.seed + round_no * 1009, slot, config["limits"], args.max_tokens))
            if not jobs:
                break
            rd = out / "rounds" / f"{round_no:04d}"
            rd.mkdir(parents=True, exist_ok=True)
            round_reasons = Counter()
            names = ["worlds.jsonl", "solutions.jsonl", "search_qa.jsonl",
                     "divergence.jsonl", "candidates.jsonl"]
            with ExitStack() as stack:
                handles = {name: stack.enter_context((rd / (name + ".tmp")).open("w", encoding="utf-8")) for name in names}
                for i, item in enumerate(pool.map(prepare_one, jobs, chunksize=1), 1):
                    key = (item["tier"], item["structure"])
                    reason = item["reason"]
                    if reason == "eligible":
                        world = env.World.from_dict(item["world"])
                        if world.group_id in group_ids:
                            reason = "duplicate_group"
                        else:
                            for other in seen[item["fingerprint"]]:
                                verdict, _ = env.isomorphic(world, other)
                                if verdict is not False:
                                    reason = "structure_duplicate" if verdict else "iso_unresolved"
                                    break
                        if reason == "eligible" and args.phase != "pilot" and accepted[key] >= target[key]:
                            reason = "quota_filled"
                        if reason == "eligible":
                            accepted[key] += 1
                            seen[item["fingerprint"]].append(world)
                            group_ids.add(world.group_id)
                            dump(handles["worlds.jsonl"], item["world"])
                            for name, rows in (("solutions.jsonl", item["solutions"]),
                                               ("search_qa.jsonl", item["records"]["search"]),
                                               ("divergence.jsonl", item["divergence"])):
                                for row in rows:
                                    dump(handles[name], row)
                    audit = {k: v for k, v in item.items() if k not in ("solutions", "records", "divergence")}
                    audit["reason"] = reason
                    dump(handles["candidates.jsonl"], audit)
                    round_reasons[reason] += 1
                    if i % 25 == 0 or i == len(jobs):
                        print(f"round={round_no} {i}/{len(jobs)} accepted={sum(accepted.values())} "
                              f"candidates={generated+i} elapsed={time.monotonic()-start:.0f}s", flush=True)
            for name in names:
                os.replace(rd / (name + ".tmp"), rd / name)
            write_json(str(rd / "complete.json"), {"candidates": len(jobs), "reasons": dict(round_reasons),
                "hashes": {name: file_sha256(str(rd / name)) for name in names}})
            generated += len(jobs)
            scheduled.update((job[0], job[1]) for job in jobs)
            reasons.update(round_reasons)
            round_no += 1
            write_json(str(out / "progress.json"), {"generated": generated, "accepted": sum(accepted.values()),
                "strata": {"/".join(k): {"accepted": accepted[k], "target": n} for k, n in target.items()},
                "reasons": dict(reasons), "elapsed_seconds": time.monotonic()-start})
    summarize(out)
    if args.phase == "build":
        if any(accepted[k] != n for k, n in target.items()):
            raise SystemExit("Candidate budget exhausted before quotas were filled; see progress.json")
        package(out, args)


def summarize(out):
    modes = defaultdict(Counter)
    strata = defaultdict(Counter)
    lengths = defaultdict(list)
    attempts = 0
    gen_reasons = Counter()
    for path in sorted((out / "rounds").glob("*/candidates.jsonl")):
        for row in env.iter_jsonl(str(path)):
            key = row["tier"] + "/" + row["structure"]
            strata[key]["candidates"] += 1
            strata[key][row["reason"]] += 1
            attempts += row["attempts"]
            gen_reasons.update(row["generation_rejections"])
            for mode, result in row.get("modes", {}).items():
                modes[row["tier"] + "/" + mode][result["status"]] += 1
                if row["reason"] == "eligible":
                    lengths[row["tier"] + "/" + mode].append(result["search_tokens"])
    write_json(str(out / "audit.json"), {"strata": dict(strata), "mode_status": dict(modes),
        "accepted_search_tokens": {k: quantiles(v) for k, v in lengths.items()},
        "generation_attempts": attempts, "generation_rejections": dict(gen_reasons),
        "truncated_training_records": 0, "external_model_calls": 0})


def stable_key(seed, world_id):
    return hashlib.sha256(f"{seed}:{world_id}".encode()).hexdigest()


def package(out, args):
    by_stratum = defaultdict(list)
    worlds = {}
    for path in sorted((out / "rounds").glob("*/worlds.jsonl")):
        for world in env.iter_jsonl(str(path)):
            worlds[world["world_id"]] = world
            by_stratum[(world["meta"]["dataset_tier"], world["meta"]["structure"])].append(world["world_id"])
    split_ids = {s: [] for s in ("train", "validation", "test")}
    train_strata = {}
    for key, ids in by_stratum.items():
        ids.sort(key=lambda wid: stable_key(args.seed, wid))
        offset = 0
        for split, n in (("validation", args.validation), ("test", args.test), ("train", args.train)):
            count = quotas(n)[key]
            chosen = ids[offset:offset+count]
            assert len(chosen) == count
            split_ids[split].extend(chosen)
            if split == "train":
                train_strata[key] = chosen
            offset += count
    id_split = {wid: split for split, ids in split_ids.items() for wid in ids}
    assert len(id_split) == len(worlds) == args.train + args.validation + args.test
    groups = {s: {worlds[wid]["group_id"] for wid in ids} for s, ids in split_ids.items()}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"])
    datasets = out / "datasets"
    datasets.mkdir(exist_ok=True)
    for split, ids in split_ids.items():
        (datasets / f"{split}.world_ids.txt").write_text("".join(wid+"\n" for wid in sorted(ids)))
        with (datasets / f"{split}.worlds.jsonl").open("w") as handle:
            for wid in sorted(ids):
                dump(handle, worlds[wid])
    subsets = datasets / "subsets"
    subsets.mkdir(exist_ok=True)
    subset_manifest = {}
    previous = set()
    for n in (1000, 2500, 5000, 10000, 15000, 20000):
        if n > args.train:
            continue
        counts = quotas(n)
        ids = {wid for key, ordered in train_strata.items() for wid in ordered[:counts[key]]}
        assert len(ids) == n and previous <= ids
        previous = ids
        path = subsets / f"train_{n}.world_ids.txt"
        path.write_text("".join(wid+"\n" for wid in sorted(ids)))
        subset_manifest[str(n)] = {"world_ids": str(path.relative_to(out)), "questions": n,
                                  "single_mode_records": n, "forward_mix_records": n*2,
                                  "four_mode_records": n*4}
    variants = {"mixed4": MODES, "forward_mixed2": search.FORWARD_MODES,
                **{mode: (mode,) for mode in MODES}}
    stats = {}
    world_tokens = defaultdict(dict)
    with ExitStack() as stack:
        handles = {}
        for split in split_ids:
            (datasets / split).mkdir(exist_ok=True)
            for view in TRAINING_VIEWS:
                for variant in variants:
                    key = (split, view, variant)
                    path = datasets / split / f"{view}_{variant}.jsonl"
                    handles[key] = stack.enter_context(Path(str(path)+".tmp").open("w"))
                    stats[key] = {"path": str(path.relative_to(out)), "records": 0, "tokens": 0,
                                  "max_tokens": 0, "distinct_targets": set(), "weight_sum": 0.0}
        for view in TRAINING_VIEWS:
            for path in sorted((out / "rounds").glob(f"*/{view}_qa.jsonl")):
                for record in env.iter_jsonl(str(path)):
                    split = id_split[record["world_id"]]
                    if view == "search":
                        world_tokens[record["world_id"]][record["source_mode"]] = record["token_lengths"]["total"]
                    for variant, modes in variants.items():
                        if record["source_mode"] not in modes:
                            continue
                        key = (split, view, variant)
                        record["target_weight"] = 1 / len(modes)
                        dump(handles[key], record)
                        stat = stats[key]
                        stat["records"] += 1
                        stat["tokens"] += record["token_lengths"]["total"]
                        stat["max_tokens"] = max(stat["max_tokens"], record["token_lengths"]["total"])
                        stat["distinct_targets"].add(record["target_hash"])
                        stat["weight_sum"] += record["target_weight"]
    final = {}
    for key, stat in stats.items():
        path = out / stat["path"]
        os.replace(str(path)+".tmp", path)
        assert stat["records"] == len(split_ids[key[0]]) * len(variants[key[2]])
        assert stat["weight_sum"] == len(split_ids[key[0]])
        stat["distinct_targets"] = len(stat["distinct_targets"])
        stat["sha256"] = file_sha256(str(path))
        stat["questions"] = len(split_ids[key[0]])
        stat["modes"] = variants[key[2]]
        final["/".join(key)] = stat
    compatible32 = {}
    ids32 = set()
    for split, ids in split_ids.items():
        kept = {wid for wid in ids if max(world_tokens[wid].values()) <= 32768}
        ids32.update(kept)
        path = datasets / f"{split}.compatible_32k.world_ids.txt"
        path.write_text("".join(wid + "\n" for wid in sorted(kept)))
        compatible32[split] = {"questions": len(kept), "world_ids": str(path.relative_to(out)),
            "strata": dict(Counter(worlds[wid]["meta"]["dataset_tier"] + "/" +
                                   worlds[wid]["meta"]["structure"] for wid in kept))}
    subsets32 = {}
    previous32 = set()
    for n in (1000, 2500, 5000, 10000, 15000, 20000):
        counts = quotas(n)
        filtered = {key: [wid for wid in ordered if wid in ids32] for key, ordered in train_strata.items()}
        if any(len(filtered[key]) < count for key, count in counts.items()):
            continue
        ids = {wid for key, ordered in filtered.items() for wid in ordered[:counts[key]]}
        assert len(ids) == n and previous32 <= ids
        previous32 = ids
        path = subsets / f"train_{n}.balanced_32k.world_ids.txt"
        path.write_text("".join(wid + "\n" for wid in sorted(ids)))
        subsets32[str(n)] = {"questions": n, "world_ids": str(path.relative_to(out))}
    with (datasets / "world_index.jsonl").open("w") as handle:
        for wid in sorted(worlds):
            assert set(world_tokens[wid]) == set(MODES)
            dump(handle, {"world_id": wid, "group_id": worlds[wid]["group_id"], "split": id_split[wid],
                 "tier": worlds[wid]["meta"]["dataset_tier"], "structure": worlds[wid]["meta"]["structure"],
                 "search_tokens": world_tokens[wid], "compatible_32k": wid in ids32})
    write_json(str(out / "manifest.json"), {"status": "complete", "task_name": args.task_name,
        "active_protocol": {"view": "search", "thinking": "on"},
        "prompt_style": qa.PROMPT_STRUCTURED, "prompt_revision": "structured-v2",
        "splits": {s: len(ids) for s, ids in split_ids.items()}, "datasets": final,
        "nested_subsets": subset_manifest, "group_overlap": 0,
        "compatible_32k": compatible32, "balanced_32k_subsets": subsets32,
        "all_accepted_records_independently_verified": True,
        "prompts_identical_across_mode_groups": True, "external_model_calls": 0,
        "note": "Nested subsets are world_id indexes into each train dataset; materialize with this script."})
    print("COMPLETE " + str(out / "manifest.json"), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("pilot", "build", "subset"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task-name", default="symbolic_qwen3_base_multimode_v1")
    p.add_argument("--tokenizer", type=Path, help="Local tokenizer directory for pilot and build")
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--validation", type=int, default=1000)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--pilot", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--max-candidates", type=int, default=110000)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-trace-lines", type=int, default=6000)
    p.add_argument("--batch-per-stratum", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--input")
    p.add_argument("--world-ids")
    args = p.parse_args()
    if args.phase == "subset":
        ids = set(Path(args.world_ids).read_text().splitlines())
        if Path(args.out).exists():
            raise SystemExit("Subset output already exists")
        with Path(args.out).open("w") as handle:
            for row in env.iter_jsonl(args.input):
                if row["world_id"] in ids:
                    dump(handle, row)
    else:
        if args.tokenizer is None:
            p.error("--tokenizer is required for pilot and build")
        build(args)


if __name__ == "__main__":
    main()

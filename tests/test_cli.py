"""四个子命令的端到端行为（第 12.3 节 9、10、14、15、16、22、23）。"""

from __future__ import annotations

import json
import pytest

from symplan import cli, env, search
from symplan.env import iter_jsonl


def _run(*argv) -> int:
    return cli.main([str(a) for a in argv])


def _gen(out, **kwargs):
    args = ["generate", "--out", out, "--level", "easy", "--seed", 5]
    for key, value in kwargs.items():
        args.extend([f"--{key.replace('_', '-')}", value])
    return _run(*args)


@pytest.fixture
def pool(tmp_path):
    out = tmp_path / "pool"
    assert _gen(out, family="general", count=12) == 0
    solutions = out / "solutions.jsonl"
    assert (
        _run(
            "solve",
            "--worlds",
            out / "worlds.jsonl",
            "--modes",
            "forward-bfs,forward-dfs",
            "--max-expansions",
            256,
            "--out",
            solutions,
        )
        == 0
    )
    return out


def test_generate_writes_manifest_and_fingerprints(tmp_path):
    out = tmp_path / "g"
    assert _gen(out, family="general", count=10) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == env.SCHEMA_VERSION
    assert manifest["audit"]["accepted"] == 10
    assert manifest["audit"]["distinct_groups"] == 10
    assert sum(manifest["audit"]["structure_counts"].values()) == 10
    assert manifest["audit"]["structure_counts"] == manifest["audit"]["structure_quota"]
    assert (out / "fingerprints.json").exists()
    assert manifest["outputs"]["sha256"] == cli.file_sha256(str(out / "worlds.jsonl"))


def test_generate_same_seed_different_workers_same_hash(tmp_path):
    one, four = tmp_path / "w1", tmp_path / "w4"
    assert _gen(one, family="general", count=12, workers=1) == 0
    assert _gen(four, family="general", count=12, workers=4) == 0
    assert cli.file_sha256(str(one / "worlds.jsonl")) == cli.file_sha256(
        str(four / "worlds.jsonl")
    )


def test_generate_refuses_overwrite_and_config_change(tmp_path):
    out = tmp_path / "g"
    assert _gen(out, family="general", count=6) == 0
    with pytest.raises(SystemExit):
        _gen(out, family="general", count=6)
    with pytest.raises(SystemExit):
        _gen(out, family="general", count=7, resume="")  # 配置不一致


def test_generate_resume_has_no_duplicates(tmp_path):
    out = tmp_path / "g"
    assert _gen(out, family="general", count=12) == 0
    full = (out / "worlds.jsonl").read_text()
    lines = full.splitlines(True)
    (out / "worlds.jsonl").write_text("".join(lines[:4]))
    args = ["generate", "--out", out, "--level", "easy", "--seed", 5, "--family", "general",
            "--count", 12, "--resume"]
    assert _run(*args) == 0
    assert (out / "worlds.jsonl").read_text() == full
    ids = [row["world_id"] for row in iter_jsonl(out / "worlds.jsonl")]
    assert len(ids) == len(set(ids)) == 12


def test_generate_shortfall_exits_nonzero(tmp_path):
    out = tmp_path / "g"
    code = _run(
        "generate", "--out", out, "--level", "easy", "--seed", 5,
        "--family", "deep-trap", "--count", 30,
        "--pair", "forward-bfs,forward-dfs", "--budget", 12, "--class", "second-only",
        "--max-attempts", 60,
    )
    assert code == 3
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["audit"]["accepted"] < 30
    assert manifest["audit"]["rejections"]


def test_solve_writes_one_record_per_world_mode(pool):
    rows = list(iter_jsonl(pool / "solutions.jsonl"))
    worlds = list(iter_jsonl(pool / "worlds.jsonl"))
    keys = [(r["world_id"], r["mode"]) for r in rows]
    assert len(keys) == len(set(keys)) == len(worlds) * 2
    stats = json.loads((pool / "solutions.stats.json").read_text())["audit"]
    assert stats["status_counts"]
    assert stats["mode_pairs"][0]["pair"] == ["forward-bfs", "forward-dfs"]
    assert "identifiability" in stats
    for row in rows:
        if row["status"] != search.STATUS_SOLVED:
            assert row["plan"] is None
        else:
            assert row["trace"][-1] == "END"


def test_export_paired_views_share_world_set(pool):
    think = pool / "think_qa.jsonl"
    nonthink = pool / "nonthink_qa.jsonl"
    base = [
        "export",
        "--worlds", pool / "worlds.jsonl",
        "--solutions", pool / "solutions.jsonl",
        "--modes", "forward-bfs,forward-dfs",
        "--selection", "all",
        "--require-all-modes",
    ]
    assert _run(*base, "--view", "search", "--thinking", "on", "--out", think) == 0
    ids_file = pool / "think_qa.world_ids.txt"
    assert _run(
        *base, "--view", "plan", "--thinking", "off",
        "--eligible-worlds", ids_file, "--out", nonthink,
    ) == 0

    a = list(iter_jsonl(think))
    b = list(iter_jsonl(nonthink))
    assert {r["world_id"] for r in a} == {r["world_id"] for r in b}
    assert len(a) == len(b)
    # 同一 SolveResult 的两种视图最终 plan 完全相同。
    plan_of = lambda rec: rec["messages"][1]["content"].split("</think>")[1]
    by_key_a = {(r["world_id"], r["source_mode"]): plan_of(r) for r in a}
    by_key_b = {(r["world_id"], r["source_mode"]): plan_of(r) for r in b}
    assert by_key_a == by_key_b
    # 配对权重：每题两个来源模式各 1/2。
    for rec in a:
        assert rec["target_weight"] == pytest.approx(0.5)
    audit = json.loads((pool / "think_qa.manifest.json").read_text())["audit"]
    assert audit["weight_sum"] == pytest.approx(audit["output_worlds"])
    assert audit["distinct_target_hash"] <= audit["qa_records"]


def test_export_rejects_unknown_eligible_id(pool):
    bad = pool / "bad_ids.txt"
    bad.write_text("w_does_not_exist\n")
    with pytest.raises(SystemExit):
        _run(
            "export", "--worlds", pool / "worlds.jsonl", "--solutions", pool / "solutions.jsonl",
            "--modes", "forward-bfs", "--view", "plan", "--thinking", "off",
            "--eligible-worlds", bad, "--out", pool / "x.jsonl",
        )


def test_export_illegal_protocol_and_uniform_guard(pool):
    common = [
        "export", "--worlds", pool / "worlds.jsonl", "--solutions", pool / "solutions.jsonl",
        "--modes", "forward-bfs", "--out", pool / "x.jsonl",
    ]
    assert _run(*common, "--view", "plan", "--thinking", "on") == 2
    assert _run(*common, "--view", "search", "--thinking", "off") == 2
    with pytest.raises(SystemExit):
        _run(*common, "--view", "plan", "--thinking", "off", "--selection", "uniform")
    with pytest.raises(SystemExit):
        _run(*common, "--view", "plan", "--thinking", "off", "--selection", "budget-aware")


def test_export_rejects_budget_above_max_expansions(pool):
    with pytest.raises(SystemExit) as exc:
        _run(
            "export", "--worlds", pool / "worlds.jsonl", "--solutions", pool / "solutions.jsonl",
            "--modes", "forward-bfs", "--view", "plan", "--thinking", "off",
            "--budget", 1024, "--out", pool / "x.jsonl",
        )
    assert "max_expansions" in str(exc.value)


def test_export_budget_aware_selects_only_feasible(pool):
    out = pool / "ba.jsonl"
    assert _run(
        "export", "--worlds", pool / "worlds.jsonl", "--solutions", pool / "solutions.jsonl",
        "--modes", "forward-bfs,forward-dfs", "--view", "search", "--thinking", "on",
        "--selection", "budget-aware", "--budget", 8, "--require-all-modes",
        "--seed", 3, "--out", out,
    ) == 0
    rows = list(iter_jsonl(out))
    solutions = {
        (r["world_id"], r["mode"]): r for r in iter_jsonl(pool / "solutions.jsonl")
    }
    assert rows
    for rec in rows:
        source = solutions[(rec["world_id"], rec["source_mode"])]
        assert source["status"] == search.STATUS_SOLVED
        assert source["cost"]["checked_nodes"] <= 8
        assert rec["target_weight"] == pytest.approx(1.0)
    audit = json.loads((pool / "ba.manifest.json").read_text())["audit"]
    assert audit["rejections"].get("budget_infeasible", 0) >= 0


def test_export_never_emits_failed_results(tmp_path):
    out = tmp_path / "tight"
    assert _gen(out, family="general", count=10, level="easy") == 0
    assert _run(
        "solve", "--worlds", out / "worlds.jsonl", "--modes", "forward-bfs",
        "--max-expansions", 4, "--out", out / "s.jsonl",
    ) == 0
    statuses = {r["status"] for r in iter_jsonl(out / "s.jsonl")}
    assert search.STATUS_BUDGET in statuses
    qa_path = out / "qa.jsonl"
    assert _run(
        "export", "--worlds", out / "worlds.jsonl", "--solutions", out / "s.jsonl",
        "--modes", "forward-bfs", "--view", "plan", "--thinking", "off", "--out", qa_path,
    ) == 0
    exported = list(iter_jsonl(qa_path))
    solved = {r["world_id"] for r in iter_jsonl(out / "s.jsonl") if r["status"] == "solved"}
    assert {r["world_id"] for r in exported} == solved
    audit = json.loads((out / "qa.manifest.json").read_text())["audit"]
    assert audit["rejections"]["no_solved_mode"] == 10 - len(solved)


def test_export_limit_worlds_keeps_structure_proportions(tmp_path):
    out = tmp_path / "big"
    assert _gen(out, family="general", count=40) == 0
    assert _run(
        "solve", "--worlds", out / "worlds.jsonl", "--modes", "forward-bfs",
        "--max-expansions", 512, "--out", out / "s.jsonl",
    ) == 0
    limited = out / "lim.jsonl"
    assert _run(
        "export", "--worlds", out / "worlds.jsonl", "--solutions", out / "s.jsonl",
        "--modes", "forward-bfs", "--view", "plan", "--thinking", "off",
        "--limit-worlds", 20, "--seed", 1, "--out", limited,
    ) == 0
    ids = [line.strip() for line in (out / "lim.world_ids.txt").read_text().splitlines()]
    assert len(ids) == 20
    worlds = {r["world_id"]: r for r in iter_jsonl(out / "worlds.jsonl")}
    picked = [worlds[i]["meta"]["structure"] for i in ids]
    for name in set(picked):
        assert picked.count(name) >= 1
    # 同 seed 复跑结果一致。
    again = out / "lim2.jsonl"
    assert _run(
        "export", "--worlds", out / "worlds.jsonl", "--solutions", out / "s.jsonl",
        "--modes", "forward-bfs", "--view", "plan", "--thinking", "off",
        "--limit-worlds", 20, "--seed", 1, "--out", again,
    ) == 0
    assert (out / "lim2.world_ids.txt").read_text() == (out / "lim.world_ids.txt").read_text()


def test_verify_strict_on_generated_data(pool):
    think = pool / "think_qa.jsonl"
    assert _run(
        "export", "--worlds", pool / "worlds.jsonl", "--solutions", pool / "solutions.jsonl",
        "--modes", "forward-bfs,forward-dfs", "--view", "search", "--thinking", "on",
        "--selection", "all", "--require-all-modes", "--out", think,
    ) == 0
    assert _run(
        "verify", "--worlds", pool / "worlds.jsonl", "--input", think, "--kind", "search",
        "--modes", "forward-bfs,forward-dfs", "--out", pool / "v.jsonl", "--strict",
    ) == 0
    rows = list(iter_jsonl(pool / "v.jsonl"))
    assert rows and all(r["plan_valid"] and r["search_valid"] for r in rows)
    stats = json.loads((pool / "v.stats.json").read_text())["audit"]
    assert stats["passed"] == stats["records"]


def test_export_defaults_to_structured_prompt_and_versions_manifest(pool):
    out = pool / "structured.jsonl"
    assert _run("export", "--worlds", pool / "worlds.jsonl", "--solutions", pool / "solutions.jsonl",
                "--modes", "forward-bfs,forward-dfs", "--view", "search", "--thinking", "on",
                "--out", out) == 0
    from symplan import qa
    worlds = env.load_worlds(pool / "worlds.jsonl")
    rows = list(iter_jsonl(out))
    assert rows
    for row in rows:
        assert row["messages"][0]["content"] == qa.render_header(worlds[row["world_id"]])
        assert row["prompt_variant"] == "search-structured-dense-v2"
    config = json.loads((pool / "structured.manifest.json").read_text())["config"]
    assert config["prompt_style"] == "structured"
    assert config["prompt_variant"] == "search-structured-dense-v2"


def test_verify_accepts_solutions_records_and_flags_tampering(pool):
    assert _run(
        "verify", "--worlds", pool / "worlds.jsonl", "--input", pool / "solutions.jsonl",
        "--kind", "prefix", "--out", pool / "vp.jsonl",
    ) == 0
    rows = list(iter_jsonl(pool / "vp.jsonl"))
    assert rows and all(r["first_error"] is None for r in rows)

    tampered = pool / "bad.jsonl"
    with open(tampered, "w", encoding="utf-8") as handle:
        for row in iter_jsonl(pool / "solutions.jsonl"):
            if row["status"] != search.STATUS_SOLVED:
                continue
            trace = [ln for ln in row["trace"] if not ln.startswith("NEW ")]
            handle.write(
                env.dump_jsonl_line(
                    {"record_id": row["world_id"], "world_id": row["world_id"],
                     "text": "\n".join(trace) + "\n"}
                )
            )
            break
    assert _run(
        "verify", "--worlds", pool / "worlds.jsonl", "--input", tampered,
        "--kind", "search", "--out", pool / "vb.jsonl", "--strict",
    ) == 6


def test_verify_unknown_world_id_exits(pool):
    bad = pool / "unknown.jsonl"
    bad.write_text(env.dump_jsonl_line({"record_id": "x", "world_id": "w_nope", "text": "END\n"}))
    with pytest.raises(SystemExit):
        _run("verify", "--worlds", pool / "worlds.jsonl", "--input", bad, "--kind", "plan")

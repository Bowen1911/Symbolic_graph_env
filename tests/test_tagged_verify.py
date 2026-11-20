"""Tagged reports join QA and verdicts without duplicate artifacts."""
import json
from dataclasses import replace

import pytest

from symplan import cli, env, qa, search, verify
from tests.fixtures import example_world


def write_rows(path, rows):
    path.write_text("".join(env.dump_jsonl_line(row) for row in rows))


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RESULT_ROOT", tmp_path / "result")
    world = example_world()
    worlds = tmp_path / "worlds.jsonl"
    write_rows(worlds, [world.to_dict(), replace(world, world_id="w_unused").to_dict()])
    predictions = tmp_path / "predictions.jsonl"
    trace = search.solve(world, "forward-bfs", search.Limits()).trace
    return world, worlds, predictions, trace


def run(worlds, predictions, *extra):
    return cli.main(["verify", "--worlds", str(worlds), "--input", str(predictions),
                     "--kind", "search", *extra])


def test_tag_reports_separate_correctness_and_coverage(inputs):
    world, worlds, predictions, trace = inputs
    cases = [trace,
             ["EXPAND n0 STATE 0 0 0 0 0", *trace[1:]],
             ["not an event", *trace[1:]],
             [*trace[:-2], "PLAN r1", "END"],
             trace[:3]]
    write_rows(predictions, [{"record_id": f"r{i}", "world_id": world.world_id,
                              "text": "\n".join(lines)} for i, lines in enumerate(cases)])
    assert run(worlds, predictions, "--tag", "qwen3_test", "--strict") == 6
    folder = cli.RESULT_ROOT / "qwen3_test"
    assert {p.name for p in folder.iterdir()} == {"qa.jsonl", "summary.json", "summary.md"}
    joined = list(env.iter_jsonl(folder / "qa.jsonl"))
    assert joined[0]["question"] == qa.render_header(world)
    assert joined[0]["model_answer"] == "\n".join(trace)
    assert joined[0]["verification"]["plan_valid"] is True
    assert joined[1]["verification"]["search_valid"] is False
    result = json.loads((folder / "summary.json").read_text())
    assert result["records"] == 5
    assert result["metrics"]["plan_valid"]["passed"] == 3
    assert result["metrics"]["syntax_complete"]["passed"] == 3
    assert result["metrics"]["passed"]["rate"] == 0.2
    assert result["metrics"]["within_budget"]["denominator"] == 0
    assert result["metrics"]["within_budget"]["rate"] is None
    assert result["metrics"]["hack"]["passed"] == 2
    assert result["primary_metric"]["name"] == "joint_success_rate"
    assert result["primary_metric"]["count"] == 1
    assert result["primary_metric"]["denominator"] == 5
    assert result["primary_metric"]["rate"] == .2
    assert result["all_records_metrics"]["all"]["counts"]["hack"] == 2
    assert result["audit"]["hack_kind"] == {"process_error": 2}
    assert result["coverage"]["worlds_with_predictions"] == 1
    assert result["coverage"]["missing_world_ids"] == ["w_unused"]
    assert result["coverage"]["attempts_per_world"] == {"5": 1}
    record = json.loads((folder / "summary.json").read_text())
    assert record["status"] == "complete" and record["exit_code"] == 6
    assert record["sources"]["input"]["sha256"] == cli.file_sha256(str(predictions))
    markdown = (folder / "summary.md").read_text()
    assert "**联合成功率：20.00%（1/5）**" in markdown
    assert markdown.index("联合成功率") < markdown.index("回答分类") < markdown.index("主要首错")
    assert "| 成功：方案正确、过程合法 | 1 | 20.00% |" in markdown
    assert "| 失败：方案正确、过程无效（hack） | 2 | 40.00% |" in markdown
    assert "最终方案正确率（辅助）：60.00%（3/5）" in markdown
    assert "问题覆盖 1/2 · 回答 5 条" in markdown
    assert "缺少 1 题回答，当前结果为部分覆盖" in markdown
    assert "截断率：未测量" in markdown
    assert "语法及包装完整" not in markdown
    assert "预算内" not in markdown
    assert not (folder / "verification.stats.json").exists()


def test_tag_preserves_inputs_and_refuses_overwrite(inputs):
    world, worlds, predictions, trace = inputs
    write_rows(predictions, [{"world_id": world.world_id, "trace": trace}])
    assert run(worlds, predictions, "--tag", "one") == 0
    folder = cli.RESULT_ROOT / "one"
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    predictions.write_text("broken json\n")
    assert run(worlds, predictions, "--tag", "one") == 2
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


@pytest.mark.parametrize("tag", ["../outside", "a/b", ".", "..", "", "/absolute", "a b"])
def test_tag_is_a_single_safe_directory(inputs, tag):
    _, worlds, predictions, _ = inputs
    assert run(worlds, predictions, "--tag", tag) == 2
    assert not cli.RESULT_ROOT.exists()


@pytest.mark.parametrize("rows", [[], [{"world_id": "w_example", "text": None}],
                                 [{"record_id": "dup", "world_id": "w_example", "text": "END"}]*2])
def test_bad_input_is_failed_run_not_zero_accuracy(inputs, rows):
    _, worlds, predictions, _ = inputs
    write_rows(predictions, rows)
    assert run(worlds, predictions, "--tag", "bad") == 2
    folder = cli.RESULT_ROOT / "bad"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert "metrics" not in summary
    assert {p.name for p in folder.iterdir()} == {"summary.json"}


def test_unknown_world_keeps_failed_report(inputs):
    _, worlds, predictions, _ = inputs
    write_rows(predictions, [{"world_id": "unknown", "text": "PLAN r0\nEND"}])
    with pytest.raises(SystemExit, match="world_id 不存在"):
        run(worlds, predictions, "--tag", "unknown")
    assert json.loads((cli.RESULT_ROOT / "unknown" / "summary.json").read_text())["status"] == "failed"


def test_tag_and_legacy_out_are_mutually_exclusive(inputs):
    _, worlds, predictions, _ = inputs
    with pytest.raises(SystemExit):
        run(worlds, predictions, "--tag", "a", "--out", "unused.jsonl")
    assert not cli.RESULT_ROOT.exists()


def test_syntax_error_does_not_hide_valid_answer(inputs):
    world, _, _, trace = inputs
    result = verify.verify_record(world, "nonsense\n"+"\n".join(trace), kind="search")
    assert result.parse_status == "error" and result.plan_valid
    assert result.search_valid is False and result.search_outcome == "invalid"
    assert result.hack and result.hack_kind == "process_error"
    duplicated = verify.verify_record(world, "nonsense\n"+"\n".join(trace)+"\nPLAN r0 r1", kind="search")
    assert duplicated.plan_valid is False


def test_unclosed_think_is_incomplete_search_but_valid_prefix(inputs):
    world, _, _, trace = inputs
    result = verify.verify_record(world, "<think>\n"+"\n".join(trace), kind="search")
    assert result.plan_valid and not result.search_valid and not result.passed
    assert result.first_error.code == env.INCOMPLETE_TRACE
    prefix = verify.verify_record(world, "<think>\n"+"\n".join(trace[:3]), kind="prefix")
    assert prefix.passed and prefix.plan_valid is None


def test_trace_protection_still_precedes_answer_execution(inputs):
    world, _, _, trace = inputs
    result = verify.verify_record(world, "\n".join(trace), kind="search", max_events=1)
    assert result.first_error.code == env.TRACE_TOO_LONG
    assert result.plan_valid is None


def test_reference_join_uses_world_id_and_preserves_raw_answers(inputs, tmp_path):
    world, worlds, predictions, trace = inputs
    answer = "<think>\n" + "\n".join(trace[:-2]) + "\n</think>\n\n" + "\n".join(trace[-2:])
    references = tmp_path / "references.jsonl"
    write_rows(references, [
        {"world_id": "w_unused", "messages": [{"role": "user", "content": "second question"},
                                               {"role": "assistant", "content": "reference B"}]},
        {"world_id": world.world_id, "messages": [{"role": "user", "content": "first question"},
                                                   {"role": "assistant", "content": "reference A"}]},
    ])
    write_rows(predictions, [
        {"world_id": world.world_id, "record_id": "a", "text": answer,
         "generated_suffix": answer[8:], "finish_reason": "stop", "completion_tokens": 42},
        {"world_id": "w_unused", "record_id": "b", "text": answer, "finish_reason": "length"},
    ])
    assert run(worlds, predictions, "--tag", "joined", "--reference", str(references)) == 0
    folder = cli.RESULT_ROOT / "joined"
    joined = list(env.iter_jsonl(folder / "qa.jsonl"))
    assert [r["question"] for r in joined] == ["first question", "second question"]
    assert [r["reference_answer"] for r in joined] == ["reference A", "reference B"]
    assert all(r["model_answer"] == answer for r in joined)
    assert joined[0]["completion_tokens"] == 42
    assert not any(k in joined[0] for k in ("text", "messages", "generated_suffix"))
    assert json.loads((folder / "summary.json").read_text())["all_records_metrics"]["all"]["truncation"]["rate"] == .5
    assert "截断率：50.00%（1/2）" in (folder / "summary.md").read_text()
    assert run(worlds, folder / "qa.jsonl", "--tag", "roundtrip") == 0
    repeated = list(env.iter_jsonl(cli.RESULT_ROOT / "roundtrip" / "qa.jsonl"))
    assert repeated == joined


def test_prefill_and_input_question_are_retained_in_qa(inputs):
    world, worlds, predictions, trace = inputs
    suffix = "\n".join(trace[:-2]) + "\n</think>\n\n" + "\n".join(trace[-2:])
    write_rows(predictions, [{"world_id": world.world_id, "question": "actual user prompt", "text": suffix}])
    assert run(worlds, predictions, "--tag", "prefill", "--prefill", "think") == 0
    row = next(env.iter_jsonl(cli.RESULT_ROOT / "prefill" / "qa.jsonl"))
    assert row["question"] == "actual user prompt"
    assert row["model_answer"] == "<think>\n" + suffix
    assert row["applied_prefill"] == "think"
    assert row["verification"]["search_valid"] is True
    summary = json.loads((cli.RESULT_ROOT / "prefill" / "summary.json").read_text())
    assert summary["all_records_metrics"]["all"]["truncation"]["rate"] is None


@pytest.mark.parametrize("duplicate", [False, True])
def test_reference_missing_or_ambiguous_is_rejected(inputs, tmp_path, duplicate):
    world, worlds, predictions, trace = inputs
    write_rows(predictions, [{"world_id": world.world_id, "trace": trace}])
    reference = tmp_path / "reference.jsonl"
    refs = [{"world_id": "w_unused", "text": "PLAN"}]
    if duplicate:
        refs = [{"world_id": world.world_id, "text": "PLAN r0"},
                {"world_id": world.world_id, "text": "PLAN r1"}]
    write_rows(reference, refs)
    assert run(worlds, predictions, "--tag", "bad_reference", "--reference", str(reference)) == 2
    folder = cli.RESULT_ROOT / "bad_reference"
    assert {p.name for p in folder.iterdir()} == {"summary.json"}
    assert json.loads((folder / "summary.json").read_text())["status"] == "failed"


def test_plan_only_has_no_joint_search_metric(inputs):
    world, worlds, predictions, trace = inputs
    write_rows(predictions, [{"world_id": world.world_id, "text": "\n".join(trace[-2:])}])
    assert cli.main(["verify", "--worlds", str(worlds), "--input", str(predictions),
                     "--kind", "plan", "--tag", "plan_only"]) == 0
    summary = json.loads((cli.RESULT_ROOT / "plan_only" / "summary.json").read_text())
    assert summary["metrics"]["passed"]["passed"] == 1
    assert summary["all_records_metrics"]["all"]["rates"]["joint_correct"] is None
    markdown = (cli.RESULT_ROOT / "plan_only" / "summary.md").read_text()
    assert "**方案通过率：100.00%（1/1）**" in markdown
    assert "联合成功" not in markdown and "hack" not in markdown.lower()
    assert "主要首错" not in markdown

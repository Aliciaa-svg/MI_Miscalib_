from __future__ import annotations

import csv
import json

import pytest

from conftest import CLEAN_CONVERSATION, make_record
from mi_miscalib.data.bucket_stratification import calibrate_thresholds, load_bucket1
from mi_miscalib.data.load_bucket import (
    build_corrupt_conversation,
    load_confidence_records,
    load_records_from_csv,
    stratify_into_buckets,
)


def test_corrupt_conversation_replaces_json_answer():
    modified, strategy = build_corrupt_conversation(CLEAN_CONVERSATION, "Berlin", "Paris")
    assert strategy == "json_answer_key"
    assert json.loads(modified[2]["content"]) == {"answer": "Paris"}
    assert CLEAN_CONVERSATION[2]["content"] == '{"answer": "Berlin"}'  # input untouched


def test_corrupt_conversation_falls_back_to_string_replace():
    raw = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": '{"answer": B}'}]
    modified, strategy = build_corrupt_conversation(raw, "B", "A")
    assert strategy == "string_predicted_answer"
    assert modified[1]["content"] == '{"answer": A}'


def test_corrupt_conversation_errors():
    with pytest.raises(ValueError):
        build_corrupt_conversation([{"role": "user", "content": "Q"}], "B", "A")
    with pytest.raises(ValueError):
        build_corrupt_conversation([{"role": "assistant", "content": "free text"}], "B", "A")


def test_stratify_into_buckets():
    records = [make_record(1, -6.0), make_record(2, 6.0), make_record(3, 0.5),
               make_record(4, -8.0, is_unknown=True)]
    buckets = stratify_into_buckets(records, -5.0, 5.0)
    assert [r.id for r in buckets[1]] == [1]
    assert [r.id for r in buckets[2]] == [2]
    assert [r.id for r in buckets[3]] == [3, 4]
    assert [r.id for r in stratify_into_buckets(records, -5.0, 5.0, exclude_unknown=False)[1]] == [1, 4]


def test_calibrate_thresholds_relaxes_until_buckets_are_large_enough():
    records = [make_record(i, -3.2) for i in range(100)] + [make_record(i, 3.2) for i in range(50)]
    assert calibrate_thresholds(records) == (-3.0, 3.0)
    assert calibrate_thresholds(records[:10]) == (-2.75, 2.75)


def _tsld_line(**fields) -> str:
    base = {"status": "success", "replacement_answer": "A", "clean_confidence": 90,
            "corrupted_confidence": 20, "delta_confidence": -70, "clean_ld_high_minus_low": 8.0,
            "corrupted_ld_high_minus_low": 1.0, "delta_high_minus_low": -7.0}
    return json.dumps({**base, **fields}) + "\n"


def _mmlu_case(subject: str, question: str, predicted: str) -> dict:
    return {
        "subject": subject, "id": 0, "question": question, "predicted_answer": predicted,
        "raw_input": [{"role": "user", "content": question},
                      {"role": "assistant", "content": f'{{"answer": {predicted}}}'}],
    }


def test_load_confidence_records_joins_mmlu_on_subject_and_id(tmp_path):
    # MMLU ids restart at 0 per subject; records must not be cross-joined.
    base = [_mmlu_case("anatomy", "Q-anatomy", "B"), _mmlu_case("virology", "Q-virology", "C")]
    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "tsld.jsonl").write_text(
        _tsld_line(subject="anatomy", id=0) + _tsld_line(subject="virology", id=0)
        + _tsld_line(subject="virology", id=0, status="failed")
    )
    records = load_confidence_records(tmp_path / "tsld.jsonl", tmp_path / "base.json")
    assert [r.question for r in records] == ["Q-anatomy", "Q-virology"]
    assert [r.corrupt_conversation[1]["content"] for r in records] == ['{"answer": A}'] * 2


def test_load_bucket1(tmp_path):
    base = [{"id": i, "question": f"Q{i}", "predicted_answer": "x",
             "raw_input": [{"role": "assistant", "content": '{"answer": "x"}'}]} for i in range(3)]
    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "tsld.jsonl").write_text(
        _tsld_line(id=0) + _tsld_line(id=1, delta_high_minus_low=6.0) + _tsld_line(id=2)
    )
    bucket1, thresholds = load_bucket1(tmp_path / "tsld.jsonl", tmp_path / "base.json", n_records=1)
    assert thresholds == (-2.75, 2.75)
    assert [r.id for r in bucket1] == [0]


def test_load_records_from_csv(tmp_path):
    path = tmp_path / "pairs.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "predicted_answer", "replacement_answer",
                         "corrupted_question", "corrupted_answer"])
        writer.writerow([7, "What is X's occupation?", "actor", "politician",
                         "What is Y's occupation?", "journalist"])
        writer.writerow([8, "Q", "", "a", "Q2", "b"])  # missing prediction -> skipped
    records = load_records_from_csv(path)
    assert len(records) == 1
    assert records[0].corrupted_answer == "journalist"
    assert "What is Y's occupation?" in records[0].corrupt_conversation[1]["content"]

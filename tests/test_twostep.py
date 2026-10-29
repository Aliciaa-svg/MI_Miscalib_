from __future__ import annotations

from mi_miscalib.data.twostep import (
    FACTUAL_STEP2_PROMPT,
    MMLU_STEP2_PROMPT,
    MMLU_SUBJECTS,
    build_step1_messages,
    first_ground_truth,
    match_free_form_answer,
    parse_choice_answer,
    parse_confidence,
    parse_free_form_answer,
    step2_prompt,
)


def test_factual_step1_prompt():
    messages = build_step1_messages({"question": "Who wrote Hamlet?"}, "popqa")
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == (
        "Question: Who wrote Hamlet?\nPlease read the question, and provide your answer as "
        "concisely as possible (usually an entity name, date, or number)\n"
        'Example: {"answer": "Paris"}'
    )


def test_mmlu_step1_prompt_lists_options():
    example = {"subject": "high_school_physics", "question": "Q?", "choices": ["a", "b", "c", "d"]}
    content = build_step1_messages(example, "mmlu")[1]["content"]
    assert content.startswith("The following is a multiple-choice question about high school physics.")
    assert "Options:\nA. a\nB. b\nC. c\nD. d\n\n\nProvide your response" in content
    assert content.endswith('{"answer": A}\n')


def test_step2_prompts():
    assert step2_prompt("popqa") == step2_prompt("nqopen") == FACTUAL_STEP2_PROMPT
    assert step2_prompt("mmlu") == MMLU_STEP2_PROMPT


def test_mmlu_has_57_subjects():
    assert len(MMLU_SUBJECTS) == len(set(MMLU_SUBJECTS)) == 57


def test_parse_free_form_answer():
    assert parse_free_form_answer('{"answer": " Paris "}') == "Paris"
    assert parse_free_form_answer('```json\n{"answer": "Paris"}\n```') == "Paris"
    assert parse_free_form_answer('{"answer": "Paris", oops') == "Paris"
    assert parse_free_form_answer("  Paris ") == "Paris"


def test_parse_choice_answer():
    assert parse_choice_answer('{"answer": "B"}') == "B"
    assert parse_choice_answer('{"answer": C}') == "C"
    assert parse_choice_answer("D. because") == "D"
    assert parse_choice_answer("no idea") is None


def test_parse_confidence():
    assert parse_confidence("85") == 85
    assert parse_confidence('{"confidence": 70}') == 70
    assert parse_confidence("Confidence: 100") == 100  # verbalized values are not clipped
    assert parse_confidence("none") is None


def test_match_free_form_answer():
    assert match_free_form_answer("the Paris", '["Paris", "City of Light"]')
    assert match_free_form_answer("Paris, France", ["Paris"])
    assert not match_free_form_answer("Berlin", ["Paris"])


def test_match_free_form_answer_apostrophe_alias_list_is_one_string():
    # Mirrors the original runs: an apostrophe breaks the quote replacement, so the
    # whole list is compared as one alias and the short alias "us" cannot match.
    gts = '["United States", "US", "Uncle Sam\'s country"]'
    assert not match_free_form_answer("Australia", gts)
    assert match_free_form_answer("Australia", '["United States", "US"]')


def test_first_ground_truth():
    assert first_ground_truth({"ground_truth": "b"}) == "B"
    assert first_ground_truth({"ground_truths": '["Paris", "City of Light"]'}) == "Paris"
    assert first_ground_truth({"ground_truths": []}) == ""

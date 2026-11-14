from ragdemo.generation import REFUSAL, SYSTEM_PROMPT, FakeGenerator, build_user_prompt, parse_answer

PASSAGES = [
    {"pid": 7, "text": "Paris is the capital of France.", "url": "https://example.org/paris"},
    {"pid": 9, "text": "The Seine flows through Paris.", "url": ""},
]


def test_prompt_numbers_passages_from_one_and_ends_with_the_question():
    prompt = build_user_prompt("What is the capital of France?", PASSAGES)
    assert "[1] Paris is the capital of France." in prompt
    assert "[2] The Seine flows through Paris." in prompt
    assert prompt.endswith("Question: What is the capital of France?")
    assert REFUSAL in SYSTEM_PROMPT


def test_parse_resolves_markers_and_counts_unresolved():
    ans = parse_answer("Paris is the capital [1]. It sits on the Seine [2][2]. Bogus [7].", PASSAGES)
    assert [c.n for c in ans.citations] == [1, 2]
    assert ans.citations[0].pid == 7 and ans.citations[0].url == "https://example.org/paris"
    assert ans.unresolved_markers == 1
    assert ans.refused is False


def test_parse_detects_refusal_exactly():
    assert parse_answer(REFUSAL, PASSAGES).refused is True
    assert parse_answer("  " + REFUSAL + "\n", PASSAGES).refused is True
    assert parse_answer("Paris [1]", PASSAGES).refused is False


def test_fake_generator_cites_first_passage_and_streams_in_order():
    gen = FakeGenerator()
    ans = gen.generate("q", PASSAGES)
    assert ans.citations[0].pid == 7 and ans.text.endswith("[1]")
    assert "".join(gen.stream("q", PASSAGES)) == ans.text
    assert gen.generate("q", []).refused is True

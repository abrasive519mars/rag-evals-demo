"""The OpenAI-compatible generator sends our prompts and parses citations like the Gemini one."""

from types import SimpleNamespace

from ragdemo.generation import REFUSAL, SYSTEM_PROMPT, OpenAIGenerator

PASSAGES = [
    {"pid": 7, "text": "Water boils at 100 degrees Celsius at sea level.", "url": "https://example.org/7"},
    {"pid": 9, "text": "Honey never spoils because of its low moisture.", "url": "https://example.org/9"},
]


class _Completions:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
                    for piece in ("Wa", "ter")
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=18),
        )


def _generator(text: str) -> tuple[OpenAIGenerator, _Completions]:
    gen = OpenAIGenerator("local", "fake-model", base_url="http://127.0.0.1:1/v1")
    completions = _Completions(text)
    gen._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return gen, completions


def test_generate_sends_prompts_and_resolves_citations():
    gen, completions = _generator("Water boils at 100 C [1]. Honey keeps [2]. Unknown [5].")
    ans = gen.generate("when does water boil", PASSAGES)
    sent = completions.calls[0]
    assert sent["model"] == "fake-model" and sent["temperature"] == 0.0
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "[1] Water boils" in sent["messages"][1]["content"]
    assert [c.pid for c in ans.citations] == [7, 9]
    assert ans.unresolved_markers == 1
    assert ans.usage == {"prompt_tokens": 120, "output_tokens": 18}
    assert ans.model == "fake-model" and not ans.refused


def test_refusal_is_detected():
    gen, _ = _generator(REFUSAL)
    assert gen.generate("who won in 2030", PASSAGES).refused


def test_stream_yields_pieces():
    gen, _ = _generator("")
    assert "".join(gen.stream("q", PASSAGES)) == "Water"

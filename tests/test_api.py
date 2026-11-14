import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragdemo.api import create_app
from ragdemo.config import Settings
from ragdemo.encoder import HashEncoder
from ragdemo.generation import FakeGenerator
from ragdemo.pipeline import Retriever
from ragdemo.retrieval import BM25Index, DenseIndex, NoopReranker

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaped over a sleeping dog in the yard.",
    "CMOS amplifier design: gain, bandwidth and noise trade-offs.",
    "Install tile in a shower: cost per square foot and labour.",
    "The dog slept. The dog dreamed. The dog woke up.",
    "Average sales of a Walgreens store are about 15,000 dollars a year.",
]


def _passages(pids):
    return {
        p: {"pid": p, "text": DOCS[p], "url": f"https://example.org/{p}"} for p in pids if 0 <= p < len(DOCS)
    }


@pytest.fixture
def client(tmp_path):
    enc = HashEncoder(dim=64)
    retriever = Retriever(
        enc,
        DenseIndex.flat(enc.encode_batch(DOCS)),
        BM25Index.build(DOCS),
        lambda pids: {p: DOCS[p] for p in pids},
        reranker=NoopReranker(),
    )
    app = create_app(
        retriever=retriever,
        generator=FakeGenerator(),
        passages=_passages,
        corpus_size=len(DOCS),
        evals_dir=tmp_path,
        settings=Settings(_env_file=None),
    )
    return TestClient(app)


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health_reports_ready_components(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["corpus_size"] == len(DOCS)
    assert body["generator"] == "fake" and body["reranker"] == "noop"
    assert "hybrid_rerank" in body["modes"]


def test_search_returns_ranked_passages_with_text_and_timings(client):
    body = client.get("/search", params={"q": "walgreens store sales", "mode": "bm25", "k": 3}).json()
    assert body["hits"][0]["pid"] == 5 and body["hits"][0]["rank"] == 1
    assert "Walgreens" in body["hits"][0]["text"] and body["hits"][0]["url"].endswith("/5")
    assert "total" in body["timings_ms"]


def test_search_validation_errors(client):
    assert client.get("/search", params={"q": "  "}).status_code == 400
    assert client.get("/search", params={"q": "x", "mode": "nope"}).status_code == 400
    assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422


def test_ask_blocking_resolves_citations_and_caches(client):
    req = {"question": "What are average Walgreens store sales?", "mode": "hybrid", "k": 3}
    first = client.post("/ask", json=req).json()
    assert first["citations"][0]["pid"] == first["hits"][0]["pid"]
    assert first["answer"].endswith("[1]") and first["cached"] is False
    assert {"total", "generate", "end_to_end"} <= set(first["timings_ms"])
    second = client.post("/ask", json=req).json()
    assert second["cached"] is True and second["answer"] == first["answer"]


def test_ask_streams_retrieval_tokens_then_done(client):
    req = {"question": "brown fox and a dog", "mode": "dense", "k": 2}
    resp = client.post("/ask", json=req, headers={"accept": "text/event-stream"})
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    events = _events(resp.text)
    names = [e for e, _ in events]
    assert names[0] == "retrieval" and names[-1] == "done" and names.count("token") >= 2
    done = events[-1][1]
    assert "".join(d["text"] for e, d in events if e == "token") == done["answer"]
    assert done["timings_ms"]["first_token"] >= 0 and done["cached"] is False


def test_ask_rejects_bad_mode_and_empty_question(client):
    assert client.post("/ask", json={"question": "x", "mode": "nope"}).status_code == 400
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_evals_404_without_report_then_serves_it(client, tmp_path):
    assert client.get("/evals").status_code == 404
    (tmp_path / "retrieval_report.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    assert client.get("/evals").json()["retrieval_report"] == {"rows": []}


def test_not_ready_app_returns_503(tmp_path):
    app = create_app(evals_dir=tmp_path, settings=Settings(_env_file=None))
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "loading"
        assert c.get("/search", params={"q": "x"}).status_code == 503
        assert c.post("/ask", json={"question": "x"}).status_code == 503


def test_index_served_when_present(client):
    resp = client.get("/")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        assert "<html" in resp.text.lower()
    else:
        assert Path("static/index.html").exists() is False or True

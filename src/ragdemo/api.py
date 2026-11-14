"""HTTP API: health, search (no LLM), ask (streamed or blocking), evals, and the single-page UI.

`create_app` takes the components explicitly so tests can inject fixtures. The module-level `app`
loads the real artifacts in a background thread at startup and reports `loading` until ready.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import SERVED_MODES, Settings, get_settings
from .generation import FakeGenerator, GeminiGenerator, Generator, parse_answer
from .pipeline import RetrievalConfig, Retriever

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    mode: str = "hybrid_rerank"
    k: int = Field(default=5, ge=1, le=20)


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever: Retriever | None = None
        self.generator: Generator | None = None
        self.passages: Callable[[list[int]], dict[int, dict]] | None = None
        self.corpus_size = 0
        self.encoder_name = settings.encoder_name
        self.error: str | None = None
        self.cache: OrderedDict[tuple, dict] = OrderedDict()
        self.cache_max = 512
        self.evals_dir = Path("evals")

    @property
    def ready(self) -> bool:
        return self.retriever is not None and self.generator is not None and self.passages is not None

    def cache_get(self, key: tuple) -> dict | None:
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
        return hit

    def cache_put(self, key: tuple, value: dict) -> None:
        self.cache[key] = value
        if len(self.cache) > self.cache_max:
            self.cache.popitem(last=False)


def _load_real(state: AppState) -> None:
    try:
        from .artifacts import SERVE_PATTERNS, Artifacts

        art = Artifacts(state.settings)
        art.ensure(SERVE_PATTERNS)
        encoder, dense, sparse = art.load_encoder(), art.load_dense(), art.load_sparse()
        dense.set_ef_search(state.settings.ef_search)
        reranker = art.load_reranker()
        state.corpus_size = art.corpus_size()
        state.passages = art.passages
        state.retriever = Retriever(encoder, dense, sparse, art.texts, reranker=reranker)
        key = state.settings.gemini_api_key
        state.generator = GeminiGenerator(key, state.settings.gen_model) if key else FakeGenerator()
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        state.error = f"{type(exc).__name__}: {exc}"


def create_app(
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    passages: Callable[[list[int]], dict[int, dict]] | None = None,
    corpus_size: int = 0,
    evals_dir: Path | None = None,
    settings: Settings | None = None,
    load_real: bool = False,
) -> FastAPI:
    state = AppState(settings or get_settings())
    state.retriever, state.generator, state.passages, state.corpus_size = (
        retriever,
        generator,
        passages,
        corpus_size,
    )
    if evals_dir is not None:
        state.evals_dir = evals_dir

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load_real and not state.ready:
            threading.Thread(target=_load_real, args=(state,), daemon=True).start()
        yield

    app = FastAPI(title="rag-evals-demo", version="0.2.0", lifespan=lifespan)
    app.state.rag = state

    def _require_ready() -> None:
        if not state.ready:
            raise HTTPException(503, detail=state.error or "artifacts are still loading")

    def _config(mode: str, k: int, k_final: int | None = None) -> RetrievalConfig:
        if mode not in SERVED_MODES:
            raise HTTPException(400, detail=f"unknown mode {mode!r}; expected one of {list(SERVED_MODES)}")
        s = state.settings
        return RetrievalConfig(
            mode=mode,
            k_candidates=max(s.k_candidates, k),
            k_final=k_final or k,
            rerank_depth=s.rerank_depth,
            ef_search=s.ef_search,
            rrf_k=s.rrf_k,
        )

    def _hits_payload(hits, passages_by_pid: dict[int, dict]) -> list[dict]:
        return [
            {
                "rank": i,
                "pid": h.pid,
                "score": round(h.score, 6),
                **passages_by_pid.get(h.pid, {"text": "", "url": ""}),
            }
            for i, h in enumerate(hits, start=1)
        ]

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if state.ready else ("error" if state.error else "loading"),
            "corpus_size": state.corpus_size,
            "modes": list(SERVED_MODES),
            "encoder": state.encoder_name,
            "reranker": state.retriever.reranker_name if state.retriever else None,
            "generator": state.generator.name if state.generator else None,
            "error": state.error,
        }

    @app.get("/search")
    def search(
        q: str = Query(default=""), mode: str = "hybrid", k: int = Query(default=10, ge=1, le=50)
    ) -> dict:
        _require_ready()
        if not q.strip():
            raise HTTPException(400, detail="query must not be empty")
        res = state.retriever.search(q, _config(mode, k))
        by_pid = state.passages([h.pid for h in res.hits])
        return {
            "query": q,
            "mode": mode,
            "hits": _hits_payload(res.hits, by_pid),
            "timings_ms": res.timings_ms,
        }

    def _retrieve(req: AskRequest) -> tuple[list[dict], list[dict], dict]:
        res = state.retriever.search(req.question, _config(req.mode, req.k))
        by_pid = state.passages([h.pid for h in res.hits])
        passages = [by_pid[h.pid] for h in res.hits if h.pid in by_pid]
        return passages, _hits_payload(res.hits, by_pid), dict(res.timings_ms)

    def _answer_payload(ans, hits, timings, cached: bool) -> dict:
        return {
            "answer": ans.text,
            "refused": ans.refused,
            "citations": [c.__dict__ for c in ans.citations],
            "unresolved_markers": ans.unresolved_markers,
            "hits": hits,
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
            "usage": ans.usage,
            "model": ans.model,
            "cached": cached,
        }

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _stream(req: AskRequest, key: tuple) -> Iterator[str]:
        t0 = time.perf_counter()
        cached = state.cache_get(key)
        if cached is not None:
            yield _sse("retrieval", {"hits": cached["hits"], "timings_ms": cached["timings_ms"]})
            yield _sse("token", {"text": cached["answer"]})
            yield _sse("done", {**cached, "cached": True})
            return
        passages, hits, timings = _retrieve(req)
        yield _sse("retrieval", {"hits": hits, "timings_ms": timings})
        chunks: list[str] = []
        first = None
        for piece in state.generator.stream(req.question, passages):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            chunks.append(piece)
            yield _sse("token", {"text": piece})
        text = "".join(chunks)
        timings["first_token"] = first or 0.0
        timings["generate"] = (time.perf_counter() - t0) * 1000 - timings.get("total", 0.0)
        timings["end_to_end"] = (time.perf_counter() - t0) * 1000
        ans = parse_answer(text, passages, model=state.generator.name)
        payload = _answer_payload(ans, hits, timings, cached=False)
        state.cache_put(key, payload)
        yield _sse("done", payload)

    @app.post("/ask")
    def ask(req: AskRequest, request: Request):
        _require_ready()
        _config(req.mode, req.k)  # validates mode
        key = (req.question.strip().lower(), req.mode, req.k)
        wants_stream = "text/event-stream" in request.headers.get("accept", "")
        if wants_stream:
            return StreamingResponse(_stream(req, key), media_type="text/event-stream")
        cached = state.cache_get(key)
        if cached is not None:
            return JSONResponse({**cached, "cached": True})
        t0 = time.perf_counter()
        passages, hits, timings = _retrieve(req)
        ans = state.generator.generate(req.question, passages)
        timings["generate"] = (time.perf_counter() - t0) * 1000 - timings.get("total", 0.0)
        timings["end_to_end"] = (time.perf_counter() - t0) * 1000
        payload = _answer_payload(ans, hits, timings, cached=False)
        state.cache_put(key, payload)
        return JSONResponse(payload)

    @app.get("/evals")
    def evals() -> dict:
        out: dict = {}
        for name in ("retrieval_report", "answers_report"):
            path = state.evals_dir / f"{name}.json"
            if path.exists():
                out[name] = json.loads(path.read_text(encoding="utf-8"))
        if not out:
            raise HTTPException(404, detail="no evaluation report yet; run `make eval`")
        return out

    @app.get("/")
    def index():
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, detail="static/index.html missing")
        return FileResponse(page)

    return app


app = create_app(load_real=True)

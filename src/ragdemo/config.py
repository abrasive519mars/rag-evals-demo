"""Runtime settings. Every field can be overridden by an environment variable of the same
name (case-insensitive) or by a .env file in the working directory."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVED_MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
EVAL_ONLY_MODES = ("dense_base", "dense_int8", "dense_100k")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # credentials
    gemini_api_key: str = ""
    openai_api_key: str = ""
    hf_token: str = ""

    # where artifacts come from and where they live locally
    hf_dataset_repo: str = "abrasive519mars/rag-evals-artifacts"
    artifacts_dir: Path = Path("data/artifacts")
    eval_dir: Path = Path("data/eval")

    # models
    encoder_name: str = "minilm-l6-msmarco-ft"  # folder inside artifacts_dir
    base_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gen_model: str = (
        "gemini-3.5-flash"  # 2.5 Flash is closed to new keys; 3.8 and 3.7 were overloaded at build time
    )
    judge_model: str = "gpt-5-mini"
    judge_fallback_model: str = "gemini-3.6-flash"  # used when the OpenAI key cannot be billed

    # retrieval defaults (see docs/design.md section 4)
    k_candidates: int = 50
    k_final: int = 5
    rerank_depth: int = 50
    ef_search: int = 64
    rrf_k: int = 60

    # serving
    port: int = 7860


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

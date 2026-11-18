"""The corpus lookup is called from the answer-generation workers and from the API's thread pool."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from ragdemo.artifacts import Artifacts
from ragdemo.config import Settings

N_PASSAGES = 500
CALLS = 6000
THREADS = 8


def _artifacts(tmp_path) -> Artifacts:
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True)
    with (art_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for pid in range(N_PASSAGES):
            f.write(
                json.dumps({"pid": pid, "text": f"passage {pid} " * 30, "url": f"https://e.org/{pid}"}) + "\n"
            )
    return Artifacts(Settings(_env_file=None, artifacts_dir=art_dir, eval_dir=tmp_path))


def test_concurrent_lookups_each_get_exactly_the_passages_they_asked_for(tmp_path):
    art = _artifacts(tmp_path)

    def lookup(i: int) -> bool:
        pids = [(i * 7 + j * 13) % N_PASSAGES for j in range(5)]
        got = art.passages(pids)
        return set(got) == set(pids) and all(got[p]["text"].startswith(f"passage {p} ") for p in pids)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(lookup, range(CALLS)))
    art.close()
    assert all(results), f"{results.count(False)} of {CALLS} lookups returned other passages"


def test_a_pid_missing_from_the_corpus_is_an_error_not_a_shorter_result(tmp_path):
    art = _artifacts(tmp_path)
    with pytest.raises(KeyError, match="not in the corpus"):
        art.passages([1, N_PASSAGES + 5])
    assert set(art.passages([1, 2])) == {1, 2}
    art.close()


def test_iter_texts_streams_in_pid_order_while_lookups_continue(tmp_path):
    art = _artifacts(tmp_path)
    texts = art.iter_texts()
    assert next(texts).startswith("passage 0 ")
    assert set(art.passages([3])) == {3}  # a scan in progress does not block a lookup
    assert next(texts).startswith("passage 1 ")
    texts.close()
    assert art.corpus_size() == N_PASSAGES
    art.close()

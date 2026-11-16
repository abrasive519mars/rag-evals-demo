"""The retrieval eval checkpoints each finished row; a rerun reuses it only for the same query count."""

from ragdemo.evals.retrieval import Row, cached_rows


def _make(name: str, n: int) -> Row:
    per_query = [{"qid": i, "recall@10": 1.0, "gold_rank": 1, "top": [i]} for i in range(n)]
    return Row(
        name=name, query_set="val1000", description="t", n=n, metrics={"recall@10": 1.0}, per_query=per_query
    )


def test_checkpoint_is_written_then_reused(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return [_make("bm25", 2)]

    first = cached_rows(tmp_path, ["bm25"], "val1000", 2, compute)
    second = cached_rows(tmp_path, ["bm25"], "val1000", 2, compute)
    assert len(calls) == 1
    assert (tmp_path / ".checkpoint" / "retrieval" / "bm25__val1000.json").exists()
    assert second[0] == first[0]
    assert second[0].per_query[1]["qid"] == 1


def test_checkpoint_with_other_query_count_is_recomputed(tmp_path):
    calls = []

    def compute(n: int):
        calls.append(1)
        return [_make("bm25", n)]

    cached_rows(tmp_path, ["bm25"], "val1000", 2, lambda: compute(2))
    cached_rows(tmp_path, ["bm25"], "val1000", 3, lambda: compute(3))
    assert len(calls) == 2


def test_group_is_recomputed_when_one_member_is_missing(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return [_make("dense_82k", 1), _make("bm25_82k", 1)]

    cached_rows(tmp_path, ["dense_82k", "bm25_82k"], "val1000", 1, compute)
    (tmp_path / ".checkpoint" / "retrieval" / "bm25_82k__val1000.json").unlink()
    cached_rows(tmp_path, ["dense_82k", "bm25_82k"], "val1000", 1, compute)
    assert len(calls) == 2

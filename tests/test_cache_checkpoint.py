"""A cache checkpoint taken while the evaluation is still writing holds only finished entries."""

import importlib.util
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ragdemo.evals.cache import write_atomic

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "colab_eval.py"


def _colab_eval():
    spec = importlib.util.spec_from_file_location("colab_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_atomic_leaves_the_entry_and_no_temporary_file(tmp_path):
    path = tmp_path / "entry.json"
    write_atomic(path, json.dumps({"text": "first"}))
    write_atomic(path, json.dumps({"text": "second"}))
    assert json.loads(path.read_text(encoding="utf-8")) == {"text": "second"}
    assert [p.name for p in tmp_path.iterdir()] == ["entry.json"]


def test_write_atomic_from_several_threads_writes_every_entry_whole(tmp_path):
    def write(i: int) -> None:
        write_atomic(tmp_path / f"{i % 50}.json", json.dumps({"n": i % 50, "pad": "x" * 2000}))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(400)))
    files = sorted(tmp_path.iterdir())
    assert len(files) == 50
    assert all(json.loads(f.read_text(encoding="utf-8"))["n"] == int(f.stem) for f in files)


def test_checkpoint_archive_round_trips_finished_entries_only(tmp_path):
    root, restored = tmp_path / "work", tmp_path / "restored"
    gen, judge = root / "data" / "cache" / "generations", root / "data" / "cache" / "judge"
    gen.mkdir(parents=True)
    judge.mkdir(parents=True)
    write_atomic(gen / "aaa.json", '{"text": "an answer"}')
    write_atomic(judge / "bbb.json", '{"verdict": "correct"}')
    (gen / "ccc.json.1234.tmp").write_text('{"text": "half wri', encoding="utf-8")  # a write in progress

    tar_path = tmp_path / "cache.tar"
    assert _colab_eval().tar_cache(root, tar_path) == 2
    with tarfile.open(tar_path) as tar:
        tar.extractall(restored, filter="data")
    found = sorted(p.relative_to(restored).as_posix() for p in restored.rglob("*") if p.is_file())
    assert found == ["data/cache/generations/aaa.json", "data/cache/judge/bbb.json"]
    assert json.loads((restored / found[0]).read_text(encoding="utf-8")) == {"text": "an answer"}

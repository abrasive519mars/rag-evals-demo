from ragdemo.config import EVAL_ONLY_MODES, SERVED_MODES, Settings


def test_settings_defaults_are_consistent():
    s = Settings(_env_file=None)
    assert s.k_final <= s.k_candidates
    assert s.rerank_depth <= s.k_candidates * 2
    assert s.rrf_k > 0


def test_modes_do_not_overlap():
    assert not set(SERVED_MODES) & set(EVAL_ONLY_MODES)

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """One retrieved passage: its row id in the corpus and the score of the stage that produced it."""

    pid: int
    score: float

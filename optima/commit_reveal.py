"""Commit-reveal + king-of-the-hill scoring — the anti-copy mechanism.

The problem in any open competition where submissions are evaluated in the open:
a lazy miner copies the current leader's submission (it's just code shipped to
the validator) and resubmits it, splitting reward for no work. Two mechanisms
defeat that here:

1. **Commit-reveal.** A miner first posts a *commitment* — a hash of
   ``(content_hash, hotkey, salt)`` — during the commit window, before any bundle
   is revealed. Later, in the reveal window, they post ``(content_hash, salt)``.
   A reveal is only accepted if it matches a commitment that *that hotkey* posted
   earlier. So you cannot reveal a bundle you didn't already commit to — and you
   couldn't have committed to a competitor's bundle you hadn't seen yet. Copying
   at reveal time is therefore impossible; the copier has no matching commitment.
   If two miners independently committed to the *same* content, the earliest
   commitment (lowest sequence) is the original; later identical ones are copies
   and earn nothing.

2. **Improvement-over-best (king of the hill).** A standing *champion* (the best
   validated bundle so far) holds the title and the emission. A challenger only
   takes the title if its score beats the champion's by a margin (which absorbs
   measurement noise). A copy ties the champion — it never clears the margin — so
   it earns zero. The only way to earn is to genuinely beat the best.

This module is pure-Python and persists to a JSON ledger so it can be tested and
reasoned about without a GPU. In a real Bittensor subnet the commitments live
on-chain, the bundles are fetched from a content-addressed store, and ``hotkey``
is the miner's SS58 address; the semantics here are the same.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


def make_commitment(content_hash: str, hotkey: str, salt: str) -> str:
    """The value a miner posts in the commit window."""
    return hashlib.sha256(f"{content_hash}:{hotkey}:{salt}".encode("utf-8")).hexdigest()


@dataclass
class Commitment:
    hotkey: str
    commitment: str
    round_id: int
    seq: int  # monotonic; commit order = anti-copy priority


@dataclass
class Reveal:
    hotkey: str
    content_hash: str
    salt: str
    round_id: int
    commit_seq: int
    original: bool = True


@dataclass
class Score:
    hotkey: str
    content_hash: str
    round_id: int
    score: float
    kl_mean: float
    passed: bool


@dataclass
class Champion:
    content_hash: str
    hotkey: str
    score: float
    round_id: int


@dataclass
class SettleResult:
    champion: Optional[Champion]
    weights: dict[str, float]
    title_changed: bool
    challenger_score: float
    rejected_copies: list[str] = field(default_factory=list)  # hotkeys


class RevealError(ValueError):
    pass


class Ledger:
    def __init__(self) -> None:
        self.commitments: list[Commitment] = []
        self.reveals: list[Reveal] = []
        self.scores: list[Score] = []
        self.champion: Optional[Champion] = None
        self._seq = 0

    # ---- persistence ----

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        led = cls()
        if not p.exists():
            return led
        data = json.loads(p.read_text())
        led.commitments = [Commitment(**c) for c in data.get("commitments", [])]
        led.reveals = [Reveal(**r) for r in data.get("reveals", [])]
        led.scores = [Score(**s) for s in data.get("scores", [])]
        champ = data.get("champion")
        led.champion = Champion(**champ) if champ else None
        led._seq = data.get("seq", len(led.commitments))
        return led

    def save(self, path: str | Path) -> None:
        data = {
            "commitments": [asdict(c) for c in self.commitments],
            "reveals": [asdict(r) for r in self.reveals],
            "scores": [asdict(s) for s in self.scores],
            "champion": asdict(self.champion) if self.champion else None,
            "seq": self._seq,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    # ---- commit phase ----

    def commit(self, hotkey: str, commitment: str, round_id: int) -> int:
        seq = self._seq
        self._seq += 1
        self.commitments.append(Commitment(hotkey, commitment, round_id, seq))
        return seq

    # ---- reveal phase ----

    def reveal(self, hotkey: str, content_hash: str, salt: str, round_id: int) -> Reveal:
        """Verify a reveal against this hotkey's prior commitments; record it.

        Raises RevealError if no commitment by this hotkey matches. Sets
        ``original`` False if an earlier-committed reveal of the same content
        already exists (a copy / duplicate).
        """
        target = make_commitment(content_hash, hotkey, salt)
        match = min(
            (c for c in self.commitments
             if c.hotkey == hotkey and c.round_id == round_id and c.commitment == target),
            key=lambda c: c.seq,
            default=None,
        )
        if match is None:
            raise RevealError(
                f"no commitment by {hotkey!r} in round {round_id} matches the revealed bundle"
            )

        # Copy detection: earliest commit_seq for this content_hash wins.
        prior = [r for r in self.reveals if r.content_hash == content_hash and r.round_id == round_id]
        original = all(match.seq < r.commit_seq for r in prior) if prior else True
        if prior and original:
            # This reveal predates earlier-recorded ones; demote them.
            for r in prior:
                r.original = False

        rev = Reveal(hotkey, content_hash, salt, round_id, match.seq, original)
        self.reveals.append(rev)
        return rev

    # ---- scoring ----

    def record_score(self, hotkey: str, content_hash: str, round_id: int,
                     score: float, kl_mean: float, passed: bool) -> None:
        self.scores.append(Score(hotkey, content_hash, round_id, score, kl_mean, passed))

    def _is_original(self, hotkey: str, content_hash: str, round_id: int) -> bool:
        for r in self.reveals:
            if r.hotkey == hotkey and r.content_hash == content_hash and r.round_id == round_id:
                return r.original
        return False

    def settle(self, round_id: int, margin: float = 0.02) -> SettleResult:
        """Apply king-of-the-hill: a challenger takes the title only if it beats
        the champion by ``margin``. Emission goes to the champion (winner-take-all
        baseline). Copies and non-improvers earn nothing.
        """
        rejected_copies: list[str] = []
        candidates: list[Score] = []
        for s in self.scores:
            if s.round_id != round_id or not s.passed:
                continue
            if not self._is_original(s.hotkey, s.content_hash, round_id):
                rejected_copies.append(s.hotkey)
                continue
            candidates.append(s)

        challenger = max(candidates, key=lambda s: s.score, default=None)
        challenger_score = challenger.score if challenger else 0.0

        title_changed = False
        threshold = (self.champion.score * (1.0 + margin)) if self.champion else (1.0 + margin)
        if challenger is not None and challenger_score >= threshold:
            self.champion = Champion(
                content_hash=challenger.content_hash,
                hotkey=challenger.hotkey,
                score=challenger.score,
                round_id=round_id,
            )
            title_changed = True

        weights = {self.champion.hotkey: 1.0} if self.champion else {}
        return SettleResult(
            champion=self.champion,
            weights=weights,
            title_changed=title_changed,
            challenger_score=challenger_score,
            rejected_copies=sorted(set(rejected_copies)),
        )

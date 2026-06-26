"""The competition agent: determinized MCTS for strategic single-select decisions,
with the greedy heuristic as a fast, robust fallback for everything else.

`agent(obs_dict) -> list[int]` is the Kaggle entry point (returns the deck when
`select is None`). `MctsAgent` is the harness-friendly object form.
"""
from __future__ import annotations

import random

from .base import BaseAgent
from .policy import choose as greedy_choose, _hard_forbidden, option_scores
from .mcts import MCTS

try:
    from cg.api import OptionType
    _END_T = OptionType.END.value
    _ATTACK_T = OptionType.ATTACK.value
except Exception:  # pragma: no cover
    _END_T, _ATTACK_T = 14, 13

# An attack the heuristic scores at/above this is a KO (every KO branch in _score_attack returns
# ~100+). Passing the turn instead of taking a guaranteed KO is never correct, so we prune END.
_KO_SCORE = 90.0
from . import config as C


def _candidates(sel: dict):
    """Enumerate full selections for a single-select decision, or None if the
    decision is multi-select (handled by the greedy policy)."""
    n = len(sel["option"])
    lo, hi = sel["minCount"], sel["maxCount"]
    if n == 0:
        return [[]]
    if hi == 1 and lo <= 1:
        cands = [[i] for i in range(n)]
        if lo == 0:
            cands.append([])
        return cands
    return None


class MctsAgent(BaseAgent):
    name = "mcts"

    def __init__(self, deck=None, seed=None, eval_fn=None, cfg=C):
        super().__init__(deck, seed)
        self.cfg = cfg
        if eval_fn is None:
            from .value_net import make_leaf_evaluator
            eval_fn = make_leaf_evaluator(cfg)
        self.mcts = MCTS(self.deck, self.rng, eval_fn=eval_fn, cfg=cfg)

    def decide(self, obs: dict) -> list[int]:
        sel = obs["select"]
        n = len(sel["option"])
        if n == 0:
            return []
        if n == 1:
            return [0] if sel["minCount"] >= 1 else greedy_choose(obs, rng=self.rng)

        cands = _candidates(sel)
        if cands is None:
            return greedy_choose(obs, rng=self.rng)  # multi-select fallback
        # Drop hard-forbidden single-select moves so the MCTS root bandit can never pick them
        # (it scores by rollout value, not the policy prior, so neutral-but-illegal-by-our-rules
        # moves like Choice Band on Meowth/Dunsparce otherwise slip through). Keep >=1 candidate.
        state = obs["current"]
        allowed = [c for c in cands if not (len(c) == 1 and _hard_forbidden(sel["option"][c[0]], state))]
        if allowed:
            cands = allowed
        # Never PASS the turn when a KO-scoring attack is available. The MCTS root bandit scores
        # by rollout value; a calibrated-but-shallow value net sometimes rates passing ~= a KO,
        # so it leaves prizes on the table (top pilots attack 64% of turns, ours 38%). Taking a
        # guaranteed KO over passing is value-net-independent and always correct, so prune END.
        opts = sel["option"]
        end_idx = [c for c in cands if len(c) == 1 and opts[c[0]].get("type") == _END_T]
        if end_idx:
            try:
                sc = option_scores(obs)
                has_ko = any(opts[i].get("type") == _ATTACK_T and sc[i] >= _KO_SCORE
                             for i in range(len(opts)))
            except Exception:
                has_ko = False
            if has_ko:
                pruned = [c for c in cands if c not in end_idx]
                if pruned:
                    cands = pruned
        try:
            pick = self.mcts.search(obs, cands)
        except Exception:
            pick = None
        if pick is None:
            return greedy_choose(obs, rng=self.rng)
        return pick


# --- Kaggle entry point ---------------------------------------------------
_AGENT: MctsAgent | None = None


def agent(obs_dict: dict) -> list[int]:
    global _AGENT
    if _AGENT is None:
        from .base import read_deck
        _AGENT = MctsAgent(deck=read_deck(), seed=random.randrange(1 << 30))
    if obs_dict.get("select") is None:
        return list(_AGENT.deck)  # initial deck selection
    return _AGENT.decide(obs_dict)

"""The competition agent: determinized MCTS for strategic single-select decisions,
with the greedy heuristic as a fast, robust fallback for everything else.

`agent(obs_dict) -> list[int]` is the Kaggle entry point (returns the deck when
`select is None`). `MctsAgent` is the harness-friendly object form.
"""
from __future__ import annotations

import random

import os

from .base import BaseAgent
from .policy import choose as greedy_choose, _hard_forbidden, option_scores
from .mcts import MCTS

try:
    from cg.api import OptionType, SelectContext
    _END_T = OptionType.END.value
    _ATTACK_T = OptionType.ATTACK.value
    _MAIN_CTX = SelectContext.MAIN.value
except Exception:  # pragma: no cover
    _END_T, _ATTACK_T, _MAIN_CTX = 14, 13, 0

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
        cands = _filter_candidates(obs, sel, cands)
        try:
            pick = self.mcts.search(obs, cands)
        except Exception:
            pick = None
        if pick is None:
            return greedy_choose(obs, rng=self.rng)
        return pick


def _filter_candidates(obs, sel, cands):
    """Drop hard-forbidden single-select moves (Choice Band on non-attackers, energy on the
    Dunsparce line) and prune END when a KO-scoring attack exists. The search core (MCTS or
    beam) scores by rollout/board value, not the policy prior, so these rule-breaks must be
    removed from the candidate set, not merely down-scored. Always keeps >=1 candidate."""
    state = obs["current"]
    opts = sel["option"]
    allowed = [c for c in cands if not (len(c) == 1 and _hard_forbidden(opts[c[0]], state))]
    if allowed:
        cands = allowed
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
    return cands


class BeamAgent(BaseAgent):
    """Own-turn forward beam search (see beam.py) — the 940+ LB rule-agent technique. Plans the
    whole turn with the engine's real forward model + a simple prize-diff eval, sidestepping the
    value-net mis-evaluation that caps the MCTS agent. Heuristic rules order the branches."""
    name = "beam"

    def __init__(self, deck=None, seed=None, cfg=C):
        super().__init__(deck, seed)
        self.cfg = cfg
        from .beam import BeamPlanner
        self.beam = BeamPlanner(self.deck, self.rng,
                                width=getattr(cfg, "BEAM_WIDTH", 3),
                                time_budget=getattr(cfg, "BEAM_TIME_BUDGET", 1.2),
                                max_depth=getattr(cfg, "BEAM_MAX_DEPTH", 8))

    def decide(self, obs: dict) -> list[int]:
        sel = obs["select"]
        n = len(sel["option"])
        if n == 0:
            return []
        if n == 1:
            return [0] if sel["minCount"] >= 1 else greedy_choose(obs, rng=self.rng)
        cands = _candidates(sel)
        if cands is None:
            return greedy_choose(obs, rng=self.rng)  # multi-select -> heuristic
        cands = _filter_candidates(obs, sel, cands)
        # Beam-search only the turn-planning MAIN decisions; sub-selections use the heuristic.
        if sel.get("context") == _MAIN_CTX:
            try:
                pick = self.beam.plan(obs, cands)
            except Exception:
                pick = None
            if pick is not None:
                return pick
        # fallback / non-MAIN: best allowed candidate by the heuristic
        g = greedy_choose(obs, rng=self.rng)
        if any(g == c for c in cands):
            return g
        return cands[0]


# --- Kaggle entry point ---------------------------------------------------
# Default to MCTS for dev; the beam-search A/B bundle sets PTCG_AGENT=beam (or patches the
# default below) so it ships the forward-beam core without touching the dev default.
_AGENT = None


def _make_agent():
    from .base import read_deck
    kind = os.environ.get("PTCG_AGENT", "mcts").lower()
    cls = BeamAgent if kind == "beam" else MctsAgent
    return cls(deck=read_deck(), seed=random.randrange(1 << 30))


def agent(obs_dict: dict) -> list[int]:
    global _AGENT
    if _AGENT is None:
        _AGENT = _make_agent()
    if obs_dict.get("select") is None:
        return list(_AGENT.deck)  # initial deck selection
    return _AGENT.decide(obs_dict)

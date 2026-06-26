"""Own-turn forward beam search — the technique the 940+ LB rule agents use.

Instead of our determinized MCTS (root bandit + value net, which mis-evaluates deeper search),
we simulate OUR OWN TURN forward with the engine's real forward model and pick the action whose
END-OF-TURN board is best, scored by a SIMPLE prize-diff heuristic (`evaluate`). No value net to
mis-calibrate; the engine itself plays the turn out, so the whole play->attach->evolve->attack
sequence is planned coherently. The hand heuristic (`option_scores`) only orders which branches
to expand, and `_hard_forbidden` prunes illegal-by-our-rules moves.

Used only for single-select MAIN decisions; everything else defers to the greedy heuristic.
"""
from __future__ import annotations

import time

from cg import api
from cg.api import to_observation_class, OptionType, SelectContext

from .mcts import _step_dict
from .determinize import determinize
from .policy import choose as policy_choose, option_scores, _hard_forbidden
from .evaluate import evaluate as eval_board

_MAIN = SelectContext.MAIN.value
_END = OptionType.END.value
_ATTACK = OptionType.ATTACK.value
_KO_SCORE = 90.0


class BeamPlanner:
    def __init__(self, deck, rng, width=4, time_budget=1.2, max_depth=8, seed_k=10, max_worlds=64):
        self.deck = deck
        self.rng = rng
        self.width = width
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.seed_k = seed_k
        self.max_worlds = max_worlds
        self.last_worlds = 0

    # --- helpers -----------------------------------------------------------
    def _ranked_main(self, obsd):
        """Heuristic-ranked option indices for a MAIN obs, hard-forbidden pruned, with END
        dropped when a KO attack is on the table (same rule as MctsAgent.decide)."""
        sel = obsd["select"]
        opts = sel.get("option") or []
        state = obsd["current"]
        try:
            sc = option_scores(obsd)
        except Exception:
            return list(range(len(opts)))
        order = sorted(range(len(opts)), key=lambda i: sc[i], reverse=True)
        order = [i for i in order if not _hard_forbidden(opts[i], state)]
        # never PASS when a KO-scoring attack exists
        if any(opts[i].get("type") == _END for i in order):
            try:
                has_ko = any(opts[i].get("type") == _ATTACK and sc[i] >= _KO_SCORE
                             for i in range(len(opts)))
            except Exception:
                has_ko = False
            if has_ko:
                order = [i for i in order if opts[i].get("type") != _END] or order
        return order or list(range(len(opts)))

    def _value(self, st, me):
        return eval_board((st.get("observation") or {}).get("current"), me)

    def _beam_one_world(self, root_sid, seeds, me, t_end):
        """Run the own-turn beam in ONE determinized world. Returns {first_action_tuple: best
        end-of-turn board value reached starting with that first action}."""
        best_by_first = {}

        def record(st, first):
            v = self._value(st, me)
            ft = tuple(first)
            if v > best_by_first.get(ft, -1e18):
                best_by_first[ft] = v
            return v

        beam = []  # (value, state_dict, first_action)
        for cand in seeds:
            try:
                st = _step_dict(root_sid, cand)
            except Exception:
                continue
            beam.append((record(st, cand), st, cand))
        if not beam:
            return best_by_first
        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[: self.width]

        depth = 0
        while depth < self.max_depth and time.perf_counter() < t_end:
            nxt = []
            advanced = False
            for val, st, first in beam:
                obsd = st.get("observation") or {}
                curd = obsd.get("current") or {}
                if curd.get("result", -1) != -1 or obsd.get("select") is None \
                        or curd.get("yourIndex") != me:
                    nxt.append((val, st, first))
                    continue
                s = obsd["select"]
                is_main_single = (s.get("context") == _MAIN and s.get("maxCount", 1) == 1
                                  and s.get("minCount", 1) <= 1)
                if not is_main_single:
                    try:
                        choice = policy_choose(obsd, rng=self.rng)
                        st2 = _step_dict(st["searchId"], choice)
                    except Exception:
                        nxt.append((val, st, first))
                        continue
                    nxt.append((record(st2, first), st2, first))
                    advanced = True
                else:
                    for oi in self._ranked_main(obsd)[: self.width]:
                        try:
                            st2 = _step_dict(st["searchId"], [oi])
                        except Exception:
                            continue
                        nxt.append((record(st2, first), st2, first))
                        advanced = True
            if not advanced:
                break
            nxt.sort(key=lambda x: x[0], reverse=True)
            beam = nxt[: self.width]
            depth += 1
        return best_by_first

    # --- main entry --------------------------------------------------------
    def plan(self, obs: dict, candidates: list[list[int]]):
        """Beam-search the rest of OUR turn over MANY determinized worlds (until the time budget),
        averaging each first action's best end-of-turn board value. Picks the first action with the
        best average. Returns None to defer to the greedy fallback."""
        me = obs["current"]["yourIndex"]
        agent_obs = to_observation_class(obs)
        t_end = time.perf_counter() + self.time_budget

        # Seed set: the heuristic's top-k candidates (saves time; weak first actions are skipped).
        sel = obs["select"]
        opts = sel.get("option") or []
        try:
            sc = option_scores(obs)
            ranked = sorted(range(len(opts)), key=lambda i: sc[i], reverse=True)
        except Exception:
            ranked = list(range(len(opts)))
        keep = set(ranked[: max(self.seed_k, self.width)])
        seeds = [c for c in candidates if (len(c) != 1) or (c[0] in keep)] or candidates

        agg = {}   # first_tuple -> [sum_value, count]
        worlds = 0
        try:
            while worlds < self.max_worlds and time.perf_counter() < t_end:
                try:
                    det = determinize(obs, self.deck, self.rng)
                    root = api.search_begin(
                        agent_obs, det.your_deck, det.your_prize,
                        det.opponent_deck, det.opponent_prize, det.opponent_hand,
                        det.opponent_active, manual_coin=False,
                    )
                    vbf = self._beam_one_world(root.searchId, seeds, me, t_end)
                except Exception:
                    if worlds == 0:
                        # search unusable for this state -> let the caller fall back
                        if not agg:
                            return None
                    break
                for ft, v in vbf.items():
                    slot = agg.setdefault(ft, [0.0, 0])
                    slot[0] += v
                    slot[1] += 1
                worlds += 1
        finally:
            try:
                api.search_end()
            except Exception:
                pass

        if not agg:
            return None
        self.last_worlds = worlds
        best_ft = max(agg, key=lambda ft: agg[ft][0] / agg[ft][1])
        return list(best_ft)

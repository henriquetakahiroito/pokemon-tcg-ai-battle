"""'What if' decision tracer — watch the Lucario rule agent think.

Plays one game (Lucario vs a chosen opponent) and, for every decision the Lucario agent makes,
records the SelectContext, each legal option decoded into plain English, the score the policy
assigned it, and which option it picked. Prints a readable turn-by-turn log and dumps the full
trace to _lucario_trace.json (for the visual viewer).

Usage:  python _trace_lucario.py [opponent_deck.csv] [max_main_decisions]
"""
import sys, json, re
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from cg.game import battle_start, battle_select
from agent.base import read_deck, BaseAgent
from agent.policy import choose as greedy_choose
from agent.lucario import (
    LucarioPolicy, LucarioAgent, AttackPlan, card_table, get_card, MEGA_BRAVE,
)
from cg.api import to_observation_class, OptionType, SelectContext, AreaType


def _name(card):
    if card is None:
        return "?"
    data = card_table.get(getattr(card, "id", -1))
    n = data.name if data else f"card{getattr(card,'id','?')}"
    return re.sub(r"[^\x20-\x7e]", "'", n).strip()


def describe(obs, opt, my_index):
    t = opt.type
    try:
        if t == OptionType.PLAY:
            return f"Play {_name(get_card(obs, AreaType.HAND, opt.index, my_index))}"
        if t == OptionType.ATTACH:
            src = get_card(obs, AreaType.HAND, opt.index, my_index)
            tgt = get_card(obs, opt.inPlayArea, opt.inPlayIndex, my_index)
            return f"Attach {_name(src)} -> {_name(tgt)} ({len(getattr(tgt,'energies',[]))}E)"
        if t == OptionType.EVOLVE:
            ev = get_card(obs, opt.area, opt.index, my_index)
            tgt = get_card(obs, opt.inPlayArea, opt.inPlayIndex, my_index)
            return f"Evolve {_name(tgt)} -> {_name(ev)}"
        if t == OptionType.ATTACK:
            tag = "Mega Brave (270)" if opt.attackId == MEGA_BRAVE else f"attack#{opt.attackId}"
            return f"Attack: {tag}"
        if t == OptionType.RETREAT:
            return "Retreat active"
        if t == OptionType.ABILITY:
            return f"Ability: {_name(get_card(obs, opt.area, opt.index, my_index))}"
        if t == OptionType.CARD:
            who = "mine" if opt.playerIndex == my_index else "opp"
            return f"Pick {_name(get_card(obs, opt.area, opt.index, opt.playerIndex))} ({who})"
        if t == OptionType.END:
            return "End turn"
        if t == OptionType.YES:
            return "Yes"
        if t == OptionType.NO:
            return "No"
        if t == OptionType.NUMBER:
            return f"Number {opt.number}"
    except Exception as e:
        return f"<{t}: {e}>"
    return f"<{t}>"


class TracingLucario(LucarioAgent):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trace = []

    def decide(self, obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            self._pre_turn = -1; self._ability_used = False; self._plan = AttackPlan()
            return list(self.deck)
        if self._pre_turn != obs.current.turn:
            self._pre_turn = obs.current.turn; self._ability_used = False; self._plan = AttackPlan()
        pol = LucarioPolicy(obs, self._plan, self._ability_used)
        if pol.context == SelectContext.MAIN:
            pol._plan_attack()
        opts = pol.select.option
        scores = [pol._score_option(o) for o in opts]
        ranked = sorted(range(len(opts)), key=lambda i: scores[i], reverse=True)
        pol._remember_lunatone_ability(ranked)
        sel = ranked[: pol.select.maxCount]
        self._plan, self._ability_used = pol.plan, pol.ability_used
        # record (skip trivial forced single-option decisions to keep the log readable)
        if len(opts) > 1:
            mi = obs.current.yourIndex
            rows = sorted(
                [{"text": describe(obs, opts[i], mi), "score": round(float(scores[i]), 1),
                  "picked": i in sel} for i in range(len(opts))],
                key=lambda r: r["score"], reverse=True)
            self.trace.append({
                "turn": obs.current.turn,
                "prizes_me": len(pol.me.prize),
                "prizes_op": len(pol.opponent.prize),
                "context": SelectContext(pol.context).name,
                "options": rows,
            })
        return sel


def main():
    opp_path = sys.argv[1] if len(sys.argv) > 1 else "deck_cand_hops_v9_clef_meowth.csv"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    LU = read_deck("deck_lucario.csv")
    OPP = read_deck(opp_path)

    class Greedy(BaseAgent):
        def decide(self, obs): return greedy_choose(obs, rng=self.rng)

    tr = TracingLucario(deck=LU, seed=7)
    opp = Greedy(deck=OPP, seed=21)
    obs, _ = battle_start(LU, OPP); plies = 0
    while obs["current"]["result"] < 0 and obs.get("select") is not None:
        who = obs["current"]["yourIndex"]
        act = tr(obs) if who == 0 else opp(obs)
        obs.pop("search_begin_input", None); obs = battle_select(act); plies += 1
        if plies > 900: break
    result = obs["current"]["result"]
    outcome = "WIN" if result == 0 else "LOSS" if result == 1 else "DRAW"

    json.dump({"opponent": opp_path, "outcome": outcome, "decisions": tr.trace},
              open("_lucario_trace.json", "w"), indent=1)

    print(f"Lucario  vs  {opp_path}   ->  {outcome}   ({plies} plies, "
          f"{len(tr.trace)} non-trivial decisions)\n")
    main_decs = [d for d in tr.trace if d["context"] == "MAIN"]
    print(f"Showing first {min(cap, len(main_decs))} MAIN (turn-planning) decisions:\n")
    for d in main_decs[:cap]:
        head = (f"--- turn {d['turn']:>2}  prizes {d['prizes_me']}-{d['prizes_op']}  "
                f"[{d['context']}] ---")
        print(head)
        for r in d["options"][:6]:
            mark = " <== PICK" if r["picked"] else ""
            print(f"    {r['score']:>8.1f}  {r['text']}{mark}")
        if len(d["options"]) > 6:
            print(f"    ... (+{len(d['options'])-6} lower-scored options)")
        print()


if __name__ == "__main__":
    main()

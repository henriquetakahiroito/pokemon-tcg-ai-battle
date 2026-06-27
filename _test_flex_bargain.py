"""Flex-slot test: Hops v10a (Lt. Surge's Bargain 1226) vs v10b (Colress's Tenacity 1194).

Both decks are identical except the 1 flex card. Pilots = our policy.py one-ply `choose`
(GreedyAgent) on BOTH seats, so Lt. Surge's Bargain IS actually played and the opponent's
prize-bargain YES/NO answer comes from our _score_yesno (defaults YES = accept).

Also instruments: how often the hero plays Bargain (card 1226), and how the opponent answers
the prize bargain (accept = YES / decline = NO) — the user's question: "is the opp greedy to accept?"

Usage: python _test_flex_bargain.py [N]
"""
import sys, os, time, collections
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"F:\Claude\pokemon-tcg-agent"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from cg.api import to_observation_class, OptionType, SelectContext

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

def deck(n): return read_deck(os.path.join(ROOT, n))

V10A = deck("deck_cand_hops_v10a_bargain.csv")
V10B = deck("deck_cand_hops_v10b_colress.csv")
LUC  = deck("deck_lucario.csv")
DRAG = deck("deck_cand_dragapult_real.csv")
ALA  = deck("deck_cand_alakazam_pro.csv")
MIRROR = deck("deck_cand_hops_v9_clef_meowth.csv")  # representative Hops mirror

OPPS = [("lucario", LUC), ("dragapult", DRAG), ("alakazam", ALA), ("mirror", MIRROR)]

# Instrumented pilot: counts (a) times THIS agent plays Lt. Surge's Bargain,
# (b) when answering a YES/NO that is NOT mulligan/first-turn (proxy for the prize bargain),
# whether it picked YES (accept) or NO.
STATS = collections.Counter()

class Probe(GreedyAgent):
    def decide(self, obs):
        try:
            o = to_observation_class(obs)
            sel = getattr(o, "select", None)
            if sel is not None:
                opts = sel.option
                ctx = sel.context
                # log a YES/NO that's a real in-game effect choice (skip mulligan + is-first)
                types = {opt.type for opt in opts}
                if {OptionType.YES, OptionType.NO} <= types and ctx not in (
                        SelectContext.MULLIGAN, SelectContext.IS_FIRST):
                    STATS["yesno_seen"] += 1
        except Exception:
            pass
        out = super().decide(obs)
        try:
            o = to_observation_class(obs)
            sel = getattr(o, "select", None)
            if sel is not None:
                for i in out:
                    if 0 <= i < len(sel.option):
                        opt = sel.option[i]
                        if opt.type == OptionType.PLAY:
                            c = o.player[o.current.yourIndex].hand[opt.index]
                            if getattr(c, "id", None) == 1226:
                                STATS["bargain_played"] += 1
                        if opt.type == OptionType.YES:
                            STATS["yesno_yes"] += 1
                        elif opt.type == OptionType.NO:
                            STATS["yesno_no"] += 1
        except Exception:
            pass
        return out

def run(hero_deck, label, instrument=False):
    print(f"--- {label} ---")
    wrs = {}
    for name, opp in OPPS:
        HeroCls = Probe if instrument else GreedyAgent
        hero = HeroCls(hero_deck, seed=1)
        oppa = GreedyAgent(opp, seed=2)
        t0 = time.perf_counter()
        res = play_match(hero, oppa, n_games=N, alternate=True)
        wrs[name] = res.winrate_a()
        print(f"  {label} vs {name:<10}: {res.wins_a:>3}W-{res.wins_b:<3}L  {res.winrate_a():.0%}  ({time.perf_counter()-t0:.0f}s)")
    return wrs

print("=" * 64)
print(f"FLEX-SLOT TEST  ({N} games/matchup, one-ply policy pilot)")
print("=" * 64)
a = run(V10A, "v10a Bargain", instrument=True)
b = run(V10B, "v10b Colress")

print("\n" + "=" * 64)
print(f"{'Matchup':<12} {'v10a Bargain':>14} {'v10b Colress':>14} {'delta':>9}")
for name, _ in OPPS:
    print(f"  {name:<10} {a[name]:>12.0%} {b[name]:>13.0%}  {(a[name]-b[name])*100:>+6.1f}pp")
print(f"  {'AVG':<10} {sum(a.values())/len(a):>12.0%} {sum(b.values())/len(b):>13.0%}  "
      f"{(sum(a.values())-sum(b.values()))/len(a)*100:>+6.1f}pp")

print("\nBARGAIN BEHAVIOR (hero v10a side):")
print(f"  Lt. Surge's Bargain played : {STATS['bargain_played']}")
print(f"  in-game YES/NO prompts seen: {STATS['yesno_seen']}")
print(f"  answered YES (accept)      : {STATS['yesno_yes']}")
print(f"  answered NO  (decline)     : {STATS['yesno_no']}")

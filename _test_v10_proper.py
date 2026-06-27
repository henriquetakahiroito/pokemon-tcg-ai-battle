"""Proper test of the finalized v10 Hops-Clefairy (Colress flex).

Compares THREE builds against the live field:
  - live   : deck.csv (current ladder build, plain Hops, no Clefairy)
  - v9      : deck_cand_hops_v9_clef_meowth.csv (old Clefairy build, the 33%-vs-Lucario trap)
  - v10     : deck_cand_hops_v10_clef.csv (NEW: -Meowth, +Dudun, 70HP Dunsparce, -1 Cramorant, +Colress)

Greedy (one-ply policy) for breadth at high N, then an MCTS (real ladder agent) confirm on the
matchups that decide the meta. Self-play is a known-weak ladder predictor — read deltas, not absolutes.

Usage: python _test_v10_proper.py [N_GREEDY] [N_MCTS]
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"F:\Claude\pokemon-tcg-agent"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match

NG = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NM = int(sys.argv[2]) if len(sys.argv) > 2 else 20

def deck(n): return read_deck(os.path.join(ROOT, n))

LIVE = deck("deck.csv")
V9   = deck("deck_cand_hops_v9_clef_meowth.csv")
V10  = deck("deck_cand_hops_v10_clef.csv")
OPPS = [("lucario", deck("deck_lucario.csv")),
        ("dragapult", deck("deck_cand_dragapult_real.csv")),
        ("alakazam", deck("deck_cand_alakazam_pro.csv"))]
# mirror = vs the current live build (so we see if v10 beats what's on the ladder now)
OPPS.append(("vs-live", LIVE))

def greedy_run(hero, label):
    print(f"--- {label} (greedy, N={NG}) ---")
    wrs = {}
    for name, opp in OPPS:
        h = GreedyAgent(hero, seed=1); o = GreedyAgent(opp, seed=2)
        t0 = time.perf_counter()
        r = play_match(h, o, n_games=NG, alternate=True)
        wrs[name] = r.winrate_a()
        print(f"  {label:<10} vs {name:<10}: {r.wins_a:>3}-{r.wins_b:<3}  {r.winrate_a():.0%}  ({time.perf_counter()-t0:.0f}s)")
    return wrs

print("=" * 66)
print("PROPER v10 TEST")
print("=" * 66)
w_live = greedy_run(LIVE, "live")
w_v9   = greedy_run(V9, "v9-clef")
w_v10  = greedy_run(V10, "v10")

print("\n" + "=" * 66)
print(f"{'Matchup':<12} {'live':>7} {'v9':>7} {'v10':>7} {'v10-live':>10}")
for name, _ in OPPS:
    print(f"  {name:<10} {w_live[name]:>6.0%} {w_v9[name]:>6.0%} {w_v10[name]:>6.0%}  {(w_v10[name]-w_live[name])*100:>+7.1f}pp")
avg = lambda w: sum(w.values())/len(w)
print(f"  {'AVG':<10} {avg(w_live):>6.0%} {avg(w_v9):>6.0%} {avg(w_v10):>6.0%}  {(avg(w_v10)-avg(w_live))*100:>+7.1f}pp")

print("\n" + "=" * 66)
print(f"MCTS CONFIRM (real ladder agent, v10 hero, N={NM})")
print("=" * 66)
from agent.agent import MctsAgent
for name, opp in [("lucario", OPPS[0][1]), ("dragapult", OPPS[1][1]), ("vs-live", LIVE)]:
    h = MctsAgent(deck=V10, seed=1); o = GreedyAgent(opp, seed=2)
    t0 = time.perf_counter()
    r = play_match(h, o, n_games=NM, alternate=True)
    print(f"  v10(MCTS) vs {name:<10}: {r.wins_a:>3}-{r.wins_b:<3}  {r.winrate_a():.0%}  ({time.perf_counter()-t0:.0f}s)")

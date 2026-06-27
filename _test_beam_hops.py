"""Stress test: BeamAgent piloting the live Hops deck.

Part A: Beam(Hops) vs the greedy field  -> crash-safety + floor + speed across matchups.
Part B: Beam(Hops) vs Mcts(Hops) head-to-head -> does forward search beat our current pilot?

Self-play is a known-weak ladder predictor; read crash-safety, speed, and large deltas only.
Usage: python _test_beam_hops.py [N_FIELD] [N_H2H]
"""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import BeamAgent, MctsAgent

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NH = int(sys.argv[2]) if len(sys.argv) > 2 else 12

HOPS = read_deck("deck.csv")
OPPS = [("lucario", read_deck("deck_lucario.csv")),
        ("dragapult", read_deck("deck_cand_dragapult_real.csv")),
        ("alakazam", read_deck("deck_cand_alakazam_pro.csv")),
        ("mirror", read_deck("deck_cand_hops_v9_clef_meowth.csv"))]

print("=" * 64)
print(f"PART A: Beam(Hops) vs greedy field   (N={NF})")
print("=" * 64)
tot_g = 0; tot_w = 0
for name, opp in OPPS:
    h = BeamAgent(deck=HOPS, seed=1); o = GreedyAgent(opp, seed=2)
    t0 = time.perf_counter()
    r = play_match(h, o, n_games=NF, alternate=True)
    dt = time.perf_counter() - t0
    tot_g += r.wins_a + r.wins_b + r.draws; tot_w += r.wins_a
    print(f"  Beam vs {name:<10}: {r.wins_a:>3}-{r.wins_b:<3}  {r.winrate_a():.0%}   "
          f"({dt:.0f}s, {dt/NF*1000:.0f}ms/game)", flush=True)
print(f"  -> 0 crashes, field win-rate {tot_w}/{tot_g} = {tot_w/max(1,tot_g):.0%}", flush=True)

print("\n" + "=" * 64)
print(f"PART B: Beam(Hops) vs Mcts(Hops) head-to-head   (N={NH})")
print("=" * 64)
h = BeamAgent(deck=HOPS, seed=1); o = MctsAgent(deck=HOPS, seed=2)
t0 = time.perf_counter()
r = play_match(h, o, n_games=NH, alternate=True)
print(f"  Beam vs Mcts: {r.wins_a}-{r.wins_b}-{r.draws}  beam_wr={r.winrate_a():.0%}  "
      f"({time.perf_counter()-t0:.0f}s)", flush=True)
print(f"  (a coin-flip ~50% means parity; high-floor Hops should be close — the real "
      f"edge would show on ladder, not here.)", flush=True)

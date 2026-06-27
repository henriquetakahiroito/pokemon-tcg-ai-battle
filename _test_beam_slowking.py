"""Stress test: BeamAgent piloting Slowking — a combo/control deck MCTS pilots ~0%.

The hypothesis from the Hops test: forward search adds no edge on high-floor proactive decks,
but COULD win on a combo deck whose payoff needs multi-step setup MCTS rarely assembles.

Part A: Beam(Slowking) vs greedy field  -> crash-safety + floor.
Part B: Beam(Slowking) vs Mcts(Slowking) head-to-head -> does forward search assemble the
        combo better than our current pilot? (If beam >> 50%, forward search is the win here.)

Usage: python _test_beam_slowking.py [N_FIELD] [N_H2H]
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import BeamAgent, MctsAgent

NF = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NH = int(sys.argv[2]) if len(sys.argv) > 2 else 12

SK = read_deck("deck_cand_slowking_combo.csv")
OPPS = [("lucario", read_deck("deck_lucario.csv")),
        ("dragapult", read_deck("deck_cand_dragapult_real.csv")),
        ("alakazam", read_deck("deck_cand_alakazam_pro.csv")),
        ("hops", read_deck("deck.csv"))]

print("=" * 64)
print(f"PART A: Beam(Slowking) vs greedy field   (N={NF})")
print("=" * 64)
tg = tw = 0
for name, opp in OPPS:
    h = BeamAgent(deck=SK, seed=1); o = GreedyAgent(opp, seed=2)
    t0 = time.perf_counter()
    r = play_match(h, o, n_games=NF, alternate=True)
    dt = time.perf_counter() - t0
    tg += r.wins_a + r.wins_b + r.draws; tw += r.wins_a
    print(f"  Beam vs {name:<10}: {r.wins_a:>3}-{r.wins_b:<3}  {r.winrate_a():.0%}   "
          f"({dt:.0f}s, {dt/NF*1000:.0f}ms/game)", flush=True)
print(f"  -> 0 crashes, field win-rate {tw}/{tg} = {tw/max(1,tg):.0%}", flush=True)

# baseline: how does MCTS pilot the SAME deck vs the same field? (memory: ~combo never assembles)
print("\n" + "=" * 64)
print(f"baseline: Mcts(Slowking) vs greedy field   (N={NF})")
print("=" * 64)
mg = mw = 0
for name, opp in OPPS:
    h = MctsAgent(deck=SK, seed=1); o = GreedyAgent(opp, seed=2)
    t0 = time.perf_counter()
    r = play_match(h, o, n_games=NF, alternate=True)
    mg += r.wins_a + r.wins_b + r.draws; mw += r.wins_a
    print(f"  Mcts vs {name:<10}: {r.wins_a:>3}-{r.wins_b:<3}  {r.winrate_a():.0%}   "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
print(f"  -> Mcts field win-rate {mw}/{mg} = {mw/max(1,mg):.0%}", flush=True)

print("\n" + "=" * 64)
print(f"PART B: Beam(Slowking) vs Mcts(Slowking) head-to-head   (N={NH})")
print("=" * 64)
h = BeamAgent(deck=SK, seed=1); o = MctsAgent(deck=SK, seed=2)
t0 = time.perf_counter()
r = play_match(h, o, n_games=NH, alternate=True)
print(f"  Beam vs Mcts: {r.wins_a}-{r.wins_b}-{r.draws}  beam_wr={r.winrate_a():.0%}  "
      f"({time.perf_counter()-t0:.0f}s)", flush=True)

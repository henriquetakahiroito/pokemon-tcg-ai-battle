"""MCTS (the real submission agent) piloting the Slowking combo deck."""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent

SLOWKING = read_deck(os.path.join(ROOT, "deck_cand_slowking_v3.csv"))
DRAG     = read_deck(os.path.join(ROOT, "deck_cand_dragapult_real.csv"))
MIRROR   = read_deck(os.path.join(ROOT, "deck_cand_hops_hybrid_v2c.csv"))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 14
print(f"MCTS Slowking vs greedy opponents, {N} games each")
for name, opp in [("Dragapult", DRAG), ("mirror", MIRROR)]:
    hero = MctsAgent(deck=SLOWKING, seed=1)
    foe  = GreedyAgent(deck=opp, seed=2)
    t0 = time.perf_counter()
    res = play_match(hero, foe, n_games=N, alternate=True)
    print(f"  MCTS Slowking vs {name:<10}: {res.wins_a}W-{res.wins_b}L  {res.winrate_a():.0%}  ({time.perf_counter()-t0:.0f}s)")

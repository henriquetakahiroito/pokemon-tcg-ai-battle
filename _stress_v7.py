"""Quick stress test: v7_clefairy (-1 Colress, -1 Clefairy vs v6) vs real ladder matchups.

Meta weights (Wave 2):
  57% Dragapult ex  17% mirror  12% Alakazam  3% Lucario
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:\Claude\pokemon-tcg-agent'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match

def deck(name):
    return read_deck(os.path.join(ROOT, name))

def mcts_agent(d, seed=1):
    from agent.agent import MctsAgent
    return MctsAgent(deck=d, seed=seed)

def greedy_agent(d, seed=1):
    return GreedyAgent(deck=d, seed=seed)

N_GREEDY = 50
N_MCTS   = 20

V7   = deck("deck_cand_hops_v7_clefairy.csv")
V6   = deck("deck_cand_hops_v6_clefairy.csv")
V2C  = deck("deck_cand_hops_hybrid_v2c.csv")
DRAG = deck("deck_cand_dragapult_real.csv")
LUC  = deck("deck_cand_lucario_riolu.csv")
ALA  = deck("deck_cand_alakazam_pro.csv")

WEIGHTS = {"dragapult": 0.57, "mirror": 0.17, "alakazam": 0.12, "lucario": 0.03}

def run_greedy(hero_deck, label, opponents):
    wrs = {}
    for name, opp_deck in opponents:
        hero = greedy_agent(hero_deck, seed=1)
        opp  = greedy_agent(opp_deck, seed=2)
        t0 = time.perf_counter()
        res = play_match(hero, opp, n_games=N_GREEDY, alternate=True)
        wr = res.winrate_a()
        wrs[name] = wr
        print(f"  {label} vs {name:<12}: {res.wins_a:>3}W-{res.wins_b:<3}L  {wr:.0%}  ({time.perf_counter()-t0:.0f}s)")
    w = sum(wrs.get(n,0)*w for n,w in WEIGHTS.items())
    print(f"  Ladder-weighted WR ({label} greedy): {w:.1%}\n")
    return wrs

opponents = [
    ("dragapult", DRAG),
    ("mirror",    V2C),
    ("lucario",   LUC),
    ("alakazam",  ALA),
]

print("=" * 60)
print(f"GREEDY SCAN  ({N_GREEDY} games/matchup)")
print("=" * 60)

print("\n--- v6_clefairy (2x Clefairy, -1 Colress, -1 Hilda) ---")
v6_wrs = run_greedy(V6, "v6", opponents)

print("--- v7_clefairy (1x Clefairy, -1 Colress, 2x Hilda) ---")
v7_wrs = run_greedy(V7, "v7", opponents)

print("=" * 60)
print("SUMMARY  (greedy)")
print("=" * 60)
print(f"{'Matchup':<14} {'v6':>6} {'v7':>6} {'delta':>8}")
for name, _ in opponents:
    g = v6_wrs.get(name, 0)
    c = v7_wrs.get(name, 0)
    print(f"  {name:<12} {g:>5.0%} {c:>5.0%}  {(c-g)*100:>+6.1f}pp")

print("\n" + "=" * 60)
print(f"MCTS vs greedy  ({N_MCTS} games each)")
print("=" * 60)
for name, opp_deck in [("dragapult", DRAG), ("mirror", V2C)]:
    hero = mcts_agent(V7, seed=1)
    opp  = greedy_agent(opp_deck, seed=2)
    t0 = time.perf_counter()
    res = play_match(hero, opp, n_games=N_MCTS, alternate=True)
    print(f"  v7(MCTS) vs {name:<12}: {res.wins_a:>3}W-{res.wins_b:<3}L  {res.winrate_a():.0%}  ({time.perf_counter()-t0:.0f}s)")

"""Stress-test v2c against real-ladder matchups using the updated MCTS agent.

Matchup weights based on real episode analysis:
  ~66% hops_hybrid mirror (v2c vs itself)
  ~23% lucario
  ~7%  alakazam
  ~4%  other

Also tests v5_brock as candidate challenger.
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

N_MCTS   = 20   # MCTS games per matchup (slow but accurate)
N_GREEDY = 50   # greedy games for broad scan

HERO_DECK = deck("deck.csv")   # v2c (restored)
CAND_DECK = deck("deck_cand_hops_v5_brock.csv")

opponents = [
    ("mirror_v2c",     deck("deck_cand_hops_hybrid_v2c.csv")),
    ("teaparty",       deck("deck_cand_teaparty.csv")),
    ("lucario",        deck("deck_cand_lucario_riolu.csv")),
    ("alakazam",       deck("deck_cand_alakazam_pro.csv")),
    ("hops_v4b",       deck("deck_cand_hops_v4b.csv")),
]

# Ladder-weighted win rate: 66% mirror + 23% lucario + 7% alakazam + 4% other
WEIGHTS = {
    "mirror_v2c":  0.66,
    "lucario":     0.23,
    "alakazam":    0.07,
    "teaparty":    0.02,
    "hops_v4b":    0.02,
}

print("=" * 60)
print("GREEDY SCAN  (v2c as hero, 50 games/matchup)")
print("=" * 60)
hero_wrs = {}
for name, opp_deck in opponents:
    hero = greedy_agent(HERO_DECK, seed=1)
    opp  = greedy_agent(opp_deck, seed=2)
    t0 = time.perf_counter()
    res = play_match(hero, opp, n_games=N_GREEDY, alternate=True)
    wr = res.winrate_a()
    hero_wrs[name] = wr
    print(f"  v2c vs {name:<15}: {res.wins_a:>3}W-{res.wins_b:<3}L  {wr:.0%}  ({time.perf_counter()-t0:.0f}s)")

w_wr = sum(hero_wrs.get(n,0)*w for n,w in WEIGHTS.items()) / sum(WEIGHTS.values())
print(f"\n  Ladder-weighted WR (v2c greedy): {w_wr:.1%}")

print("\n" + "=" * 60)
print("MCTS STRESS  (v2c vs mirror + lucario + alakazam, 20 games)")
print("=" * 60)
mcts_wrs = {}
for name, opp_deck in opponents[:3]:   # mirror, teaparty, lucario
    hero = mcts_agent(HERO_DECK, seed=1)
    opp  = greedy_agent(opp_deck, seed=2)
    t0 = time.perf_counter()
    res = play_match(hero, opp, n_games=N_MCTS, alternate=True)
    wr = res.winrate_a()
    mcts_wrs[name] = wr
    print(f"  v2c(MCTS) vs {name:<12}: {res.wins_a:>3}W-{res.wins_b:<3}L  {wr:.0%}  ({time.perf_counter()-t0:.0f}s)")

print("\n" + "=" * 60)
print("CHALLENGER: v5_brock greedy vs same opponents")
print("=" * 60)
cand_wrs = {}
for name, opp_deck in opponents:
    hero = greedy_agent(CAND_DECK, seed=1)
    opp  = greedy_agent(opp_deck, seed=2)
    t0 = time.perf_counter()
    res = play_match(hero, opp, n_games=N_GREEDY, alternate=True)
    wr = res.winrate_a()
    cand_wrs[name] = wr
    print(f"  v5 vs {name:<15}: {res.wins_a:>3}W-{res.wins_b:<3}L  {wr:.0%}  ({time.perf_counter()-t0:.0f}s)")

w_wr_c = sum(cand_wrs.get(n,0)*w for n,w in WEIGHTS.items()) / sum(WEIGHTS.values())
print(f"\n  Ladder-weighted WR (v5 greedy): {w_wr_c:.1%}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Matchup':<18} {'v2c greedy':>12} {'v5 greedy':>10} {'delta':>8}")
for name, _ in opponents:
    g = hero_wrs.get(name, 0)
    c = cand_wrs.get(name, 0)
    print(f"  {name:<16} {g:>11.0%} {c:>9.0%}  {(c-g)*100:>+6.1f}pp")
print(f"\n  Ladder-weighted: v2c={w_wr:.1%}  v5={w_wr_c:.1%}  delta={(w_wr_c-w_wr)*100:+.1f}pp")
if w_wr_c > w_wr + 0.03:
    print("  => v5 is materially better — consider switching main deck")
elif w_wr_c < w_wr - 0.02:
    print("  => v2c holds the edge — keep it")
else:
    print("  => within noise — v2c remains the safe choice")

"""MCTS gauntlet: v3 Slowking (Mimikyu) vs the top leaderboard archetypes.

Each matchup flushes immediately so partial results are readable. Two seeds on the
most important matchups to gauge consistency (not just a single lucky run).
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent

SLOWKING = read_deck(os.path.join(ROOT, "deck_cand_slowking_v3.csv"))
OPPS = {
    "teaparty":  "deck_cand_teaparty.csv",       # #1 leaderboard (Debauchery Tea Party)
    "dragapult": "deck_cand_dragapult_real.csv",  # BDIF, 57% of meta
    "mirror":    "deck_cand_hops_hybrid_v2c.csv",
    "alakazam":  "deck_cand_alakazam_pro.csv",
    "lucario":   "deck_cand_lucario_riolu.csv",
}
# (name, deck, hero_seed, n_games) — two seeds on the marquee matchups for consistency
PLAN = [
    ("teaparty",  1, 16),
    ("dragapult", 1, 16),
    ("dragapult", 7, 16),   # consistency check, different seed
    ("teaparty",  7, 16),   # consistency check
    ("alakazam",  1, 14),
    ("lucario",   1, 14),
    ("mirror",    7, 14),   # consistency check (seed 1 was already 100%)
]

print(f"MCTS gauntlet: v3 Slowking (Mimikyu) vs top decks\n{'='*56}")
agg = {}
for name, seed, n in PLAN:
    opp = read_deck(os.path.join(ROOT, OPPS[name]))
    hero = MctsAgent(deck=SLOWKING, seed=seed)
    foe = GreedyAgent(deck=opp, seed=seed + 100)
    t0 = time.perf_counter()
    res = play_match(hero, foe, n_games=n, alternate=True)
    wr = res.winrate_a()
    agg.setdefault(name, []).append((res.wins_a, res.wins_b, wr, seed))
    print(f"  vs {name:<10} (seed {seed:>2}): {res.wins_a:>2}W-{res.wins_b:<2}L  {wr:.0%}  ({time.perf_counter()-t0:.0f}s)")

print(f"\n{'='*56}\nSUMMARY (consistency across seeds)")
for name, runs in agg.items():
    total_w = sum(r[0] for r in runs); total_g = sum(r[0]+r[1] for r in runs)
    print(f"  {name:<10}: " + "  ".join(f"{r[2]:.0%}(s{r[3]})" for r in runs) +
          f"   | combined {total_w}/{total_g} = {total_w/total_g:.0%}")

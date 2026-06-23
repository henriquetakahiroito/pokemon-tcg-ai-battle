"""Same gauntlet as _mcts_gauntlet.py but for the current primary deck (v7 Clefairy),
so v3 Slowking vs v7 Clefairy is an apples-to-apples comparison (same opps + seeds)."""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent

HERO = read_deck(os.path.join(ROOT, "deck_cand_hops_v7_clefairy.csv"))
OPPS = {
    "teaparty":  "deck_cand_teaparty.csv",
    "dragapult": "deck_cand_dragapult_real.csv",
    "mirror":    "deck_cand_hops_hybrid_v2c.csv",
    "alakazam":  "deck_cand_alakazam_pro.csv",
    "lucario":   "deck_cand_lucario_riolu.csv",
}
PLAN = [
    ("teaparty",  1, 16),
    ("dragapult", 1, 16),
    ("dragapult", 7, 16),
    ("teaparty",  7, 16),
    ("alakazam",  1, 14),
    ("lucario",   1, 14),
    ("mirror",    7, 14),
]
print(f"MCTS gauntlet: v7 Clefairy (current primary) vs top decks\n{'='*56}")
agg = {}
for name, seed, n in PLAN:
    opp = read_deck(os.path.join(ROOT, OPPS[name]))
    hero = MctsAgent(deck=HERO, seed=seed)
    foe = GreedyAgent(deck=opp, seed=seed + 100)
    t0 = time.perf_counter()
    res = play_match(hero, foe, n_games=n, alternate=True)
    agg.setdefault(name, []).append((res.wins_a, res.wins_b, res.winrate_a(), seed))
    print(f"  vs {name:<10} (seed {seed:>2}): {res.wins_a:>2}W-{res.wins_b:<2}L  {res.winrate_a():.0%}  ({time.perf_counter()-t0:.0f}s)")
print(f"\n{'='*56}\nSUMMARY (v7 Clefairy)")
for name, runs in agg.items():
    tw = sum(r[0] for r in runs); tg = sum(r[0]+r[1] for r in runs)
    print(f"  {name:<10}: " + "  ".join(f"{r[2]:.0%}(s{r[3]})" for r in runs) + f"   | combined {tw}/{tg} = {tw/tg:.0%}")

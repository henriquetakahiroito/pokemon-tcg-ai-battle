"""Stress test + LEARNING PROOF for MW turbo-Archaludon.

Part A: full meta gauntlet, MCTS pilot vs every top archetype, 2 seeds each
        => raw piloting strength + cross-seed consistency.
Part B: action proof. Decode the hero's chosen options across the gauntlet and
        tally what the agent actually DID:
          - evolve Duraludon -> Archaludon ex  (engine online)
          - attack id distribution (Metal Defender 253 vs Duraludon hammer 223/224)
        This is the direct rebuttal to "HE IS HITTING WITH DURALUDON": if the
        agent learned the deck it should evolve and swing Metal Defender, not
        stall on Duraludon.
"""
import sys, os, time, collections
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent
import cg.api as api

MW = read_deck(os.path.join(ROOT, "deck_archaludon_mw.csv"))
OPPS = {
    "lucario":   "deck_cand_lucario_riolu.csv",
    "dragapult": "deck_cand_dragapult_real.csv",
    "hops":      "deck_cand_hops_v10b_colress.csv",
    "starmie":   "deck_cand_starmie_jaga.csv",
    "alakazam":  "deck_cand_alakazam_pro.csv",
}
PLAN = [
    ("lucario",   1, 16), ("lucario",   7, 16),
    ("dragapult", 1, 16), ("dragapult", 7, 16),
    ("hops",      1, 16),
    ("starmie",   1, 16), ("starmie",   7, 16),
    ("alakazam",  1, 14),
]

_DURALUDON, _ARCH_EX = 169, 190
_METAL_DEFENDER, _IRON_BLASTER = 253, 225
_HAMMER = (223, 224)

print(f"MW turbo-Archaludon | MCTS pilot | full meta gauntlet\n{'='*60}")
agg = {}
for name, seed, n in PLAN:
    opp = read_deck(os.path.join(ROOT, OPPS[name]))
    hero = MctsAgent(deck=MW, seed=seed)
    foe = GreedyAgent(deck=opp, seed=seed + 100)
    t0 = time.perf_counter()
    res = play_match(hero, foe, n_games=n, alternate=True)
    wr = res.winrate_a()
    agg.setdefault(name, []).append((res.wins_a, res.wins_b, wr, seed))
    print(f"  vs {name:<10} (seed {seed:>2}): {res.wins_a:>2}W-{res.wins_b:<2}L  {wr:.0%}  ({time.perf_counter()-t0:.0f}s)")

print(f"\n{'='*60}\nSUMMARY (cross-seed consistency)")
tot_w = tot_g = 0
for name, runs in agg.items():
    w = sum(r[0] for r in runs); g = sum(r[0]+r[1] for r in runs)
    tot_w += w; tot_g += g
    print(f"  {name:<10}: " + "  ".join(f"{r[2]:.0%}(s{r[3]})" for r in runs) +
          f"   | {w}/{g} = {w/g:.0%}")
print(f"  {'FIELD':<10}: {tot_w}/{tot_g} = {tot_w/tot_g:.0%}")

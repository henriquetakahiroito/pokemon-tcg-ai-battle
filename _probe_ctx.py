"""Capture which SelectContext fires when the Slowking deck places cards on the
deck (Academy at Night / Ciphermaniac), so we know where to inject combo logic.

Logs, for seat 0, every non-MAIN/standard context and the card ids of its options.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)

from cg.api import OptionType, SelectContext
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose, _card_id_from_option
from agent.base import read_deck
from agent.cards import get_db
import random
from collections import Counter

db = get_db()
SLOWKING = read_deck(os.path.join(ROOT, "deck_cand_slowking_v2.csv"))
OPP = read_deck(os.path.join(ROOT, "deck_cand_dragapult_real.csv"))

# Contexts we care about (deck/look/order placement)
WATCH = {9: "TO_DECK", 10: "TO_DECK_BOTTOM", 12: "NOT_MOVE", 24: "LOOK",
         25: "EFFECT_TARGET", 7: "TO_HAND", 34: "SKILL_ORDER"}

ctx_seen = Counter()
samples = {}   # ctx_name -> list of option-id lists (first few)

def greedy(deck, seed):
    r = random.Random(seed)
    def f(obs):
        if obs.get("select") is None: return list(deck)
        return choose(obs, rng=r)
    return f

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
a = greedy(SLOWKING, 1)
b = greedy(OPP, 2)
for g in range(N):
    obs, start = battle_start(SLOWKING, OPP)
    if obs is None: continue
    agents = (a, b)
    steps = 0
    try:
        while True:
            state = obs.get("current")
            if state and state.get("result", -1) != -1: break
            sel = obs.get("select")
            if sel is None: break
            who = state["yourIndex"]
            if who == 0:
                ctx = sel["context"]
                if ctx in WATCH:
                    name = WATCH[ctx]
                    ctx_seen[name] += 1
                    if name not in samples:
                        samples[name] = []
                    if len(samples[name]) < 6:
                        ids = []
                        for o in sel["option"]:
                            cid = _card_id_from_option(o, state)
                            nm = db.name(cid) if cid else f"type{o.get('type')}"
                            ids.append(f"{nm}({cid})")
                        samples[name].append(ids)
            obs = battle_select(agents[who](obs))
            steps += 1
            if steps > 30000: break
    finally:
        battle_finish()

print("=" * 60)
print(f"Contexts seat-0 encountered (over {N} games):")
print("=" * 60)
for name, cnt in ctx_seen.most_common():
    print(f"\n  {name}: {cnt} times")
    for s in samples.get(name, []):
        print(f"     options: {s}")

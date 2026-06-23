"""Decisive probe for the Mimikyu anti-Dragapult line:

   Slowking Seek Inspiration -> copies Mimikyu's Gemstone Mimicry
   -> copies Dragapult's ACTIVE Phantom Dive (200)
   -> with Lillie's Clefairy benched (Fairy Zone), Slowking is Psychic -> x2 = 400 -> OHKO 320 HP.

We need to confirm the engine resolves this NESTED copy. A tuned forcer drives toward it
and we watch the opponent's ACTIVE hp for a big drop / KO right after a Seek that discarded Mimikyu.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)
from cg.api import OptionType
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose, _card_id_from_option
from agent.base import read_deck
from agent.cards import get_db
import random

db = get_db()
_DECK_FILE = sys.argv[2] if len(sys.argv) > 2 else "deck_cand_slowking_v3.csv"
SLOWKING = read_deck(os.path.join(ROOT, _DECK_FILE))
DRAG = read_deck(os.path.join(ROOT, "deck_cand_dragapult_real.csv"))
SEEK, MIMICRY, PHANTOM = 213, 612, 154
MIMIKYU_ID, CLEFAIRY_ID = 434, 272

def active_hp(p):
    a = p.get("active") or []
    return (a[0].get("hp", 0) or 0) if (a and a[0]) else 0

def active_id(p):
    a = p.get("active") or []
    return a[0].get("id") if (a and a[0]) else None

class Forcer:
    """Drive toward the Mimikyu line; greedy elsewhere."""
    def __init__(self, deck, seed):
        self.deck = deck; self.rng = random.Random(seed)
    def __call__(self, obs):
        sel = obs.get("select")
        if sel is None: return list(self.deck)
        state = obs["current"]; opts = sel["option"]
        ctx = sel["context"]
        lo, hi = sel.get("minCount", 1), min(sel.get("maxCount", 1), len(opts))
        single = lo <= 1 <= hi
        # 1) prefer copy-chain attacks: Seek > Mimicry > Phantom Dive
        if single:
            for want in (SEEK, MIMICRY, PHANTOM):
                for i, o in enumerate(opts):
                    if o.get("type") == OptionType.ATTACK.value and o.get("attackId") == want:
                        return [i]
        # 2) TO_DECK from hand: put Mimikyu on top when Dragapult is active
        if ctx == 9 and single:
            op = state["players"][1 - state["yourIndex"]]
            if active_id(op) is not None:
                for i, o in enumerate(opts):
                    if _card_id_from_option(o, state) == MIMIKYU_ID:
                        return [i]
        return choose(obs, rng=self.rng)

def greedy(deck, seed):
    r = random.Random(seed)
    def f(obs):
        if obs.get("select") is None: return list(deck)
        return choose(obs, rng=r)
    return f

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
print(f"Mimikyu line probe: {N} games, Slowking (forcer) vs Dragapult (greedy)")
seek_uses = 0; mimikyu_discarded = 0; big_hits = []; kos = 0
dragapult_active_at_fire = 0; fire_active_ids = []
a = Forcer(SLOWKING, 1); b = greedy(DRAG, 2)
for g in range(N):
    obs, start = battle_start(SLOWKING, DRAG)
    if obs is None: continue
    agents = (a, b); pending = None; steps = 0
    try:
        while True:
            st = obs.get("current")
            if st and st.get("result", -1) != -1: break
            sel = obs.get("select")
            if sel is None: break
            who = st["yourIndex"]
            op = st["players"][1]
            if pending is not None and who == 1:
                hp_b, did_b, oactive_b = pending
                d = [ (c or {}).get("id") for c in (st["players"][0].get("discard") or []) ]
                top = d[-1] if len(d) > did_b else None
                if top == MIMIKYU_ID:
                    mimikyu_discarded += 1
                    fire_active_ids.append(oactive_b)
                    if oactive_b == 121:  # Dragapult ex was the active Tera at fire time
                        dragapult_active_at_fire += 1
                    drop = hp_b - active_hp(op)
                    # if opp active changed/removed => KO
                    if active_id(op) != oactive_b or active_hp(op) == 0:
                        kos += 1
                    if drop > 0:
                        big_hits.append(drop)
                pending = None
            choice = agents[who](obs)
            if who == 0:
                for idx in choice:
                    if 0 <= idx < len(sel["option"]):
                        o = sel["option"][idx]
                        if o.get("type") == OptionType.ATTACK.value and o.get("attackId") == SEEK:
                            seek_uses += 1
                            pending = (active_hp(op), len(st["players"][0].get("discard") or []), active_id(op))
            obs = battle_select(choice)
            steps += 1
            if steps > 30000: break
    finally:
        battle_finish()

print()
print("=" * 60)
from collections import Counter
print(f"  Seek Inspiration fired         : {seek_uses}")
print(f"  ...discarded Mimikyu off top   : {mimikyu_discarded}")
print(f"  ...with Dragapult ACTIVE       : {dragapult_active_at_fire}   <-- Mimicry only works vs active Tera")
print(f"  opp active ids at Mimikyu fire : {dict(Counter(db.name(i) if i else None for i in fire_active_ids))}")
print(f"  ...opp ACTIVE hp drops (Phantom): {sorted(big_hits, reverse=True)[:10]}")
print(f"  ...resulting KO of opp active  : {kos}")
print()
if any(h >= 320 for h in big_hits) or kos:
    print("  => NESTED COPY + Fairy Zone OHKO CONFIRMED working in engine.")
elif any(h >= 150 for h in big_hits):
    print("  => Nested copy works (Phantom Dive landed); Fairy Zone x2 may or may not have applied.")
elif mimikyu_discarded:
    print("  => Mimikyu was copied but no big damage observed — check Tera/weakness handling.")
else:
    print("  => line never set up in this sample (need more games or MCTS).")

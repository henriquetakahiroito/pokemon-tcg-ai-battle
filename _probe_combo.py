"""Measure whether the REAL greedy policy (not a forcer) executes the Slowking combo:
sets a payoff on top, fires Seek Inspiration, and resolves a copied attack.

Reports win rate + combo execution stats vs Dragapult and vs a Hops mirror.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)

from cg.api import OptionType
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose
from agent.base import read_deck
from agent.cards import get_db
import random

db = get_db()
SLOWKING = read_deck(os.path.join(ROOT, "deck_cand_slowking_v2.csv"))
SEEK, TRIFROST, AXE, DESTINED = 213, 188, 785, 305
PAYOFF_ATTACKS = {TRIFROST: "Trifrost", AXE: "Axe Blast", DESTINED: "Destined Fight"}

def board_hp(p):
    t = 0
    a = p.get("active") or []
    if a and a[0]: t += a[0].get("hp", 0) or 0
    for b in (p.get("bench") or []):
        if b: t += b.get("hp", 0) or 0
    return t

def greedy(deck, seed):
    r = random.Random(seed)
    def f(obs):
        if obs.get("select") is None: return list(deck)
        return choose(obs, rng=r)
    return f

def run(opp_deck, opp_name, n, seed0=1):
    wins = 0; games = 0
    seek_uses = 0; seek_effect = 0; payoff_copies = 0
    copied = {}
    a = greedy(SLOWKING, seed0)
    b = greedy(opp_deck, seed0 + 500)
    for g in range(n):
        # alternate seats for fairness
        a_seat0 = (g % 2 == 0)
        s0, s1 = (a, b) if a_seat0 else (b, a)
        my_seat = 0 if a_seat0 else 1
        obs, start = battle_start(s0.deck if hasattr(s0, 'deck') else SLOWKING,
                                  s1.deck if hasattr(s1, 'deck') else opp_deck)
        if obs is None: continue
        agents = (s0, s1); pending = None; steps = 0
        try:
            while True:
                st = obs.get("current")
                if st and st.get("result", -1) != -1:
                    if st["result"] == my_seat: wins += 1
                    break
                sel = obs.get("select")
                if sel is None: break
                who = st["yourIndex"]
                opp_idx = 1 - my_seat
                # resolve pending Seek effect when control returns to opponent
                if pending is not None and who == opp_idx:
                    hp_before, disc_before = pending
                    hp_now = board_hp(st["players"][opp_idx])
                    if hp_before - hp_now > 0:
                        seek_effect += 1
                        d = [ (c or {}).get("id") for c in (st["players"][my_seat].get("discard") or []) ]
                        top = d[-1] if len(d) > disc_before else None
                        copied[top] = copied.get(top, 0) + 1
                    pending = None
                choice = agents[who](obs)
                if who == my_seat:
                    for idx in choice:
                        if 0 <= idx < len(sel["option"]):
                            o = sel["option"][idx]
                            if o.get("type") == OptionType.ATTACK.value:
                                if o.get("attackId") == SEEK:
                                    seek_uses += 1
                                    pending = (board_hp(st["players"][opp_idx]),
                                               len(st["players"][my_seat].get("discard") or []))
                                elif o.get("attackId") in PAYOFF_ATTACKS:
                                    payoff_copies += 1
                obs = battle_select(choice)
                steps += 1
                if steps > 30000: break
        finally:
            battle_finish()
        games += 1
    wr = wins / games if games else 0
    print(f"\n  vs {opp_name}: {wins}/{games} = {wr:.0%}")
    print(f"     Seek Inspiration fired : {seek_uses}")
    print(f"     ...with opp HP loss    : {seek_effect}")
    print(f"     copied PAYOFF attack   : {payoff_copies}  (Trifrost/Axe Blast/Destined Fight chosen)")
    if copied:
        print("     copied top-card breakdown:")
        for cid, cnt in sorted(copied.items(), key=lambda x: -x[1]):
            print(f"        {db.name(cid) if cid else '(none)':<26} x{cnt}")
    return wr

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    DRAG = read_deck(os.path.join(ROOT, "deck_cand_dragapult_real.csv"))
    MIRROR = read_deck(os.path.join(ROOT, "deck_cand_hops_hybrid_v2c.csv"))
    print("=" * 60)
    print(f"REAL greedy policy — Slowking combo execution ({N} games each)")
    print("=" * 60)
    run(DRAG, "Dragapult", N)
    run(MIRROR, "Hops mirror", N)

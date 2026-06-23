"""Decisive probe: does the engine resolve Slowking 'Seek Inspiration' attack-copy?

Strategy: a forcer agent (seat 0, Slowking deck) always picks Seek Inspiration (213)
when available, else greedy. We watch player[1]'s board HP and player[0]'s prize
count across the turn. If, right after a Seek Inspiration, the opponent loses HP or
we take a prize, the copied attack resolved in the engine.

We also read the card discarded off the top (last entry added to player[0] discard)
to see WHICH non-Rule-Box attack got copied.
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
SLOWKING_DECK = read_deck(os.path.join(ROOT, "deck_cand_slowking_v2.csv"))
OPP_DECK = read_deck(os.path.join(ROOT, "deck_cand_dragapult_real.csv"))
SEEK = 213
SUPER_PSY = 214

def board_hp(player):
    tot = 0
    a = player.get("active") or []
    if a and a[0]: tot += a[0].get("hp", 0) or 0
    for b in (player.get("bench") or []):
        if b: tot += b.get("hp", 0) or 0
    return tot

def discard_ids(player):
    return [ (c or {}).get("id") for c in (player.get("discard") or []) ]

class Forcer:
    def __init__(self, deck, seed):
        self.deck = deck; self.rng = random.Random(seed)
        self.seek_idx_chosen = False
    def __call__(self, obs):
        sel = obs.get("select")
        if sel is None:
            return list(self.deck)
        opts = sel["option"]
        # Prefer Seek Inspiration if it's an available attack
        for i, o in enumerate(opts):
            if o.get("type") == OptionType.ATTACK.value and o.get("attackId") == SEEK:
                if sel.get("minCount", 1) <= 1 <= min(sel.get("maxCount", 1), len(opts)):
                    return [i]
        return choose(obs, rng=self.rng)

def greedy(deck, seed):
    r = random.Random(seed)
    def f(obs):
        if obs.get("select") is None: return list(deck)
        return choose(obs, rng=r)
    return f

def run_games(n, seed0=1):
    seek_uses = 0
    seek_with_effect = 0          # opp HP dropped or prize taken right after
    psy_uses = 0
    psy_with_effect = 0
    copied = {}                   # discarded top-card id -> count of effect resolutions
    forcer = Forcer(SLOWKING_DECK, seed0)
    opp = greedy(OPP_DECK, seed0 + 500)
    games_played = 0
    wins = 0
    for g in range(n):
        seat0, seat1 = forcer, opp
        obs, start = battle_start(seat0.deck if hasattr(seat0,'deck') else SLOWKING_DECK, OPP_DECK)
        if obs is None:
            continue
        agents = (seat0, seat1)
        pending = None   # (hp_before_p1, prize0_before, discard0_len_before)
        steps = 0
        try:
            while True:
                state = obs.get("current")
                if state and state.get("result", -1) != -1:
                    if state["result"] == 0: wins += 1
                    break
                sel = obs.get("select")
                if sel is None: break
                who = state["yourIndex"]
                p1 = state["players"][1]
                p0 = state["players"][0]
                # Resolve a pending Seek/Psy measurement once control returns to seat 1
                if pending is not None and who == 1:
                    kind, hp_before, prize_before, disc_before = pending
                    hp_now = board_hp(p1)
                    prize_now = len(p0.get("prize") or [])
                    delta_hp = hp_before - hp_now
                    prize_taken = prize_before - prize_now
                    effect = (delta_hp > 0) or (prize_taken > 0)
                    if kind == "seek":
                        if effect:
                            seek_with_effect += 1
                            d_now = discard_ids(p0)
                            top = d_now[-1] if len(d_now) > disc_before else None
                            copied[top] = copied.get(top, 0) + 1
                    else:
                        if effect: psy_with_effect += 1
                    pending = None
                # Decide
                choice = agents[who](obs)
                # Detect what seat 0 is about to do
                if who == 0 and pending is None:
                    for idx in choice:
                        if 0 <= idx < len(sel["option"]):
                            o = sel["option"][idx]
                            if o.get("type") == OptionType.ATTACK.value:
                                if o.get("attackId") == SEEK:
                                    seek_uses += 1
                                    pending = ("seek", board_hp(p1), len(p0.get("prize") or []), len(p0.get("discard") or []))
                                elif o.get("attackId") == SUPER_PSY:
                                    psy_uses += 1
                                    pending = ("psy", board_hp(p1), len(p0.get("prize") or []), len(p0.get("discard") or []))
                obs = battle_select(choice)
                steps += 1
                if steps > 30000: break
        finally:
            battle_finish()
        games_played += 1
    return dict(games=games_played, wins=wins, seek_uses=seek_uses,
                seek_with_effect=seek_with_effect, psy_uses=psy_uses,
                psy_with_effect=psy_with_effect, copied=copied)

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"Running {N} games: Slowking (forcer, seat0) vs Dragapult (greedy)...")
    r = run_games(N)
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  games played            : {r['games']}")
    print(f"  forcer wins             : {r['wins']}")
    print(f"  Seek Inspiration used   : {r['seek_uses']}")
    print(f"  ...with board effect    : {r['seek_with_effect']}   <-- copy mechanic works if > 0")
    print(f"  Super Psy Bolt used     : {r['psy_uses']}  (control)")
    print(f"  ...with board effect    : {r['psy_with_effect']}   <-- detection sanity if > 0")
    print()
    print("  Copied top-card -> effect-resolution count:")
    if r['copied']:
        for cid, cnt in sorted(r['copied'].items(), key=lambda x: -x[1]):
            nm = db.name(cid) if cid is not None else "(unknown/none)"
            print(f"     {nm:<28} (id={cid}): {cnt}")
    else:
        print("     (none observed)")

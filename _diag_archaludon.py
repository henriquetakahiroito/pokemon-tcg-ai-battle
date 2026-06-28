"""Pilot diagnostic: which cards does the Archaludon pilot never play, and is its score map sane?

Plays N games (our policy as both seats) and, for every decision WE make, records each option's
resolved card id, option type, the policy score, and whether it was picked. Aggregates per card:
  - offered / played counts (PLAY+EVOLVE+ATTACH+ABILITY+ATTACK = "used")
  - score min/avg/max when offered
Then flags deck cards NEVER used, and prints the full score map to eyeball mistakes.

Usage: python _diag_archaludon.py [N] [deck.csv]
"""
import sys, random, collections
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from cg.game import battle_start, battle_select
from agent.base import read_deck
from agent.cards import get_db
import agent.policy as P
from agent.policy import option_scores, _card_id_from_option, choose
from cg.api import OptionType

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
DECKF = sys.argv[2] if len(sys.argv) > 2 else "deck_turbo_archaludon.csv"
db = get_db()
DECK = read_deck(DECKF)
OPPS = ["deck_lucario.csv", "deck_cand_starmie.csv", "deck_cand_alakazam_pro.csv",
        "deck.csv", "deck_cand_dragapult_real.csv"]

USED_TYPES = {OptionType.PLAY.value, OptionType.EVOLVE.value, OptionType.ATTACH.value,
              OptionType.ABILITY.value, OptionType.ATTACK.value}

# per (card_id) -> stats; and per (card_id, type) -> score samples
offered = collections.Counter(); used = collections.Counter()
scoremap = collections.defaultdict(list)   # (cid, typename) -> [scores when offered]
picked_scores = collections.defaultdict(list)

def typename(t):
    try: return OptionType(t).name
    except Exception: return str(t)

for g in range(N):
    opp = read_deck(OPPS[g % len(OPPS)])
    obs, _ = battle_start(DECK, opp); plies = 0
    rng = random.Random(g)
    while obs["current"]["result"] < 0 and obs.get("select") is not None and plies < 900:
        who = obs["current"]["yourIndex"]
        if who == 0:
            try:
                scores = option_scores(obs)
                sel = obs["select"]; opts = sel["option"]; state = obs["current"]
                # which indices does choose() pick?
                act = choose(obs, rng=rng)
                pickset = set(act)
                for i, o in enumerate(opts):
                    t = o.get("type")
                    if t not in USED_TYPES:
                        continue
                    cid = _card_id_from_option(o, state)
                    if cid is None:
                        # ATTACK options resolve to the active attacker
                        if t == OptionType.ATTACK.value:
                            a = (state["players"][0].get("active") or [{}])
                            cid = (a[0] or {}).get("id") if a else None
                    if cid is None:
                        continue
                    key = (cid, typename(t))
                    offered[cid] += 1
                    scoremap[key].append(round(float(scores[i]), 1))
                    if i in pickset:
                        used[cid] += 1
                        picked_scores[key].append(round(float(scores[i]), 1))
                obs.pop("search_begin_input", None); obs = battle_select(act); plies += 1
                continue
            except Exception as e:
                pass
        # opponent or fallback
        act = choose(obs, rng=random.Random(99)) if who == 1 else choose(obs, rng=rng)
        obs.pop("search_begin_input", None); obs = battle_select(act); plies += 1

deck_ids = set(DECK)
def nm(cid):
    try: return db.card(cid).name
    except Exception: return "?"

print(f"=== {DECKF}: {N} games, our-side decisions instrumented ===\n")
print("CARDS IN DECK NEVER USED (offered but never played, or never even drawn into a decision):")
for cid in sorted(deck_ids):
    if used[cid] == 0:
        tag = f"offered {offered[cid]}x, played 0" if offered[cid] else "never reached a decision"
        print(f"   {cid:<5} {nm(cid):<22} {tag}")

print("\nSCORE MAP (per card x option-type): offered / played, score min..avg..max")
rows = []
for (cid, tn), scs in scoremap.items():
    o = len(scs); u = len(picked_scores[(cid, tn)])
    rows.append((cid, tn, o, u, min(scs), sum(scs)/len(scs), max(scs)))
for cid, tn, o, u, mn, av, mx in sorted(rows, key=lambda r: (r[0], -r[2])):
    flag = "  <-- never picked" if u == 0 else ""
    print(f"   {cid:<5} {nm(cid):<20} {tn:<8} off={o:<4} play={u:<4} "
          f"score {mn:>6.1f}..{av:>6.1f}..{mx:>6.1f}{flag}")

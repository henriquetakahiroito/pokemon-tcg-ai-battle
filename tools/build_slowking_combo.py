"""Slowking + Kyurem + Academy at Night toolbox deck.

User-specified archetype using Slowking's Seek Inspiration to copy attacks
from non-Rule-Box Pokemon placed on top of deck via Academy at Night.
Toolbox of attack options:
  * Kyurem Trifrost (110x3 board sweep) — main combo
  * Annihilape Destined Fight — 1-prize trade for major KOs
  * Spectrier Phantasmal Barrage — anti-bench-protection (Shaymin counter)

Substitutions vs user spec (per Henrique's tuning):
  * Lillie's Clefairy ex CUT (Dragapult/Raging Bolt aren't on the Kaggle ladder)
  * 1 Special Red Card -> 1 Xerosic's Machinations (caps opp hand at 3 — kills
    Powerful Hand AND mirror draw engines)
  * +1 Slowking (3 -> 4) — chain reliability with freed slot
  * +1 Basic Psychic Energy (user's listing summed to 59 instead of 60)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.cards import get_db, validate_deck

DECK = [
    # Pokemon (19) — Clefairy CUT, +1 Slowking
    (144, 2),   # Kyurem (Trifrost target)
    (184, 1),   # Latias ex (Skyliner = free retreat)
    (276, 1),   # Metagross
    (162, 4),   # Slowpoke
    (140, 1),   # Fezandipiti ex
    (163, 4),   # Slowking (Seek Inspiration)  [3 -> 4]
    (434, 1),   # Team Rocket's Mimikyu
    (1071, 1),  # Meowth ex
    (183, 1),   # Smoochum (energizes Slowpoke)
    (224, 1),   # Annihilape (Destined Fight)
    (880, 1),   # Spectrier (Phantasmal Barrage anti-Shaymin)
    (235, 1),   # Budew (turn 1 disruption)
    (343, 1),   # Shaymin (Flower Curtain own bench protection)
    # Trainers (31)
    (1227, 3),  # Lillie's Determination
    (1092, 1),  # Secret Box (ACE SPEC)
    (1225, 1),  # Hilda
    (1188, 1),  # Ciphermaniac's Codebreaking
    (1248, 4),  # Academy at Night (stadium)
    (1231, 1),  # Dawn
    (1123, 1),  # Switch
    (1182, 2),  # Boss's Orders
    (1184, 1),  # Lana's Aid
    (1097, 4),  # Night Stretcher
    (1156, 1),  # Lucky Helmet
    (1121, 3),  # Ultra Ball
    (1152, 4),  # Poke Pad
    (1194, 1),  # Colress's Tenacity
    (1146, 2),  # Wondrous Patch
    (1197, 1),  # Xerosic's Machinations (substitute for Special Red Card)
    # Energy (10)
    (19, 3),    # Telepathic Psychic Energy
    (9, 3),     # Boomerang Energy
    (5, 3),     # Basic Psychic Energy
]


def main():
    db = get_db()
    ids = []
    for cid, n in DECK:
        ids.extend([cid] * n)
    ok, probs = validate_deck(ids)
    ne = sum(1 for c in ids if db.is_energy(c))
    print(f"slowking_combo: cards={len(ids)} energy={ne}  -> {'OK' if ok else 'ILLEGAL '+str(probs)}")
    if ok:
        out = os.path.join(ROOT, "deck_cand_slowking_combo.csv")
        with open(out, "w", newline="") as f:
            f.write("\n".join(str(c) for c in ids) + "\n")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()

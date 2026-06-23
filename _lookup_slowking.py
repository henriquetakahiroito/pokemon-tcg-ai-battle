"""Look up card IDs + attack details for the Slowking/Kyurem archetype."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db
from cg.api import EnergyType

db = get_db()

# Names to find (substring match, case-insensitive)
wanted = [
    "Slowpoke", "Slowking", "Mega Kangaskhan", "Kangaskhan", "Latias",
    "Kyurem", "Metagross", "Meowth", "Zeraora", "Lillie's Clefairy",
    "Fezandipiti", "Lillie's Determination", "Ciphermaniac", "Codebreaking",
    "Poké Pad", "Poke Pad", "Ultra Ball", "Wondrous Patch", "Night Stretcher",
    "Secret Box", "Switch", "Brave Bangle", "Lucky Helmet", "Academy at Night",
    "Telepathic", "Psychic Energy", "Boomerang",
]

print("=" * 70)
print("CARD ID LOOKUP")
print("=" * 70)
all_cards = db.all_cards()
seen = {}
for w in wanted:
    matches = []
    for cid, c in all_cards.items():
        nm = (c.name or "")
        if w.lower() in nm.lower():
            matches.append((cid, nm, c))
    if matches:
        for cid, nm, c in sorted(matches, key=lambda x: x[0]):
            key = (cid, nm)
            if key in seen:
                continue
            seen[key] = True
            etype = ""
            try:
                etype = EnergyType(c.energyType).name if c.energyType is not None else ""
            except Exception:
                etype = str(c.energyType)
            extra = []
            if getattr(c, "ex", False): extra.append("EX")
            if getattr(c, "basic", False): extra.append("basic")
            if getattr(c, "aceSpec", False): extra.append("ACE")
            from cg.api import CardType
            try:
                ctype = CardType(c.cardType).name
            except Exception:
                ctype = str(c.cardType)
            hp = getattr(c, "hp", "")
            print(f"  [{cid:>5}] {nm:<34} {ctype:<14} {etype:<10} HP={hp} {' '.join(extra)}")
    else:
        print(f"  (no match) {w}")

print()
print("=" * 70)
print("ATTACK DETAILS — Slowking, Kyurem, Metagross, Mega Kangaskhan, Zeraora")
print("=" * 70)

def show_attacks(name_sub):
    for cid, c in sorted(all_cards.items()):
        nm = c.name or ""
        if name_sub.lower() in nm.lower():
            atks = db.attacks_of(cid)
            if not atks and not (c.abilities if hasattr(c, 'abilities') else None):
                continue
            print(f"\n  [{cid}] {nm} (HP={getattr(c,'hp','')}):")
            # abilities
            ab = getattr(c, "abilities", None)
            if ab:
                print(f"      abilities raw: {ab}")
            for a in atks:
                energies = [EnergyType(e).name for e in a.energies]
                print(f"      ATK [{a.attackId}] {a.name}: dmg={a.damage}, cost={a.cost} {energies}")
                if a.text:
                    print(f"           text: {a.text}")

for n in ["Slowking", "Kyurem", "Metagross", "Mega Kangaskhan", "Zeraora", "Slowpoke", "Lillie's Clefairy"]:
    show_attacks(n)

print()
print("=" * 70)
print("STADIUM / KEY TRAINER TEXT — Academy at Night, Wondrous Patch, Ciphermaniac")
print("=" * 70)
for n in ["Academy at Night", "Wondrous Patch", "Ciphermaniac", "Brave Bangle", "Lucky Helmet", "Telepathic", "Boomerang"]:
    for cid, c in sorted(all_cards.items()):
        nm = c.name or ""
        if n.lower() in nm.lower():
            txt = getattr(c, "text", "") or getattr(c, "rulesText", "") or ""
            print(f"\n  [{cid}] {nm}:")
            if txt:
                print(f"      {txt}")
            else:
                # dump all string attributes
                for attr in dir(c):
                    if attr.startswith("_"): continue
                    val = getattr(c, attr, None)
                    if isinstance(val, str) and len(val) > 15:
                        print(f"      {attr}: {val}")

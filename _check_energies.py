from agent.cards import get_db
db = get_db()

# Get text/details for Mist Energy, Legacy Energy, Telepath Psychic Energy
for cid in [11, 12, 19]:
    c = db.card(cid)
    print(f"id={cid} name={db.name(cid)}")
    if c:
        print(f"  text: {getattr(c, 'text', 'N/A')}")
        print(f"  energyType: {getattr(c, 'energyType', 'N/A')}")

# Check Trevenant's Horrifying Revenge full text
from cg.api import all_attack
for a in all_attack():
    if a.attackId in (1267, 1268):
        print(f"\nAttack {a.attackId} '{a.name}': dmg={a.damage} text='{a.text}'")

# Check Alakazam card
print("\n=== Cards with 'Alakazam' ===")
for cid, c in db.all_cards().items():
    if c.name and "alakazam" in c.name.lower():
        print(f"  id={cid} name={c.name}")
        atks = db.attacks_of(cid)
        for a in atks:
            print(f"    attack {a.attackId} '{a.name}' dmg={a.damage} text='{a.text[:100]}'")

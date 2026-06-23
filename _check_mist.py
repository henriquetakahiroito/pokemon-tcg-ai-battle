from agent.cards import get_db
db = get_db()

# Check current deck cards
deck_ids = [1086, 1152, 1122, 1115, 1171, 1227, 1182, 1225, 1194, 1097, 1255, 19, 11, 12]
print("=== Current deck non-Pokemon cards ===")
for cid in deck_ids:
    c = db.card(cid)
    if c:
        print(f"  id={cid} name={db.name(cid)} type={c.cardType}")

# Search for "Mist" in all cards
print("\n=== Cards with 'Mist' in name ===")
for cid, c in db.all_cards().items():
    if c.name and "mist" in c.name.lower():
        print(f"  id={cid} name={c.name} type={c.cardType}")

# Search for special energies that might protect from effects
print("\n=== Special energy cards ===")
from cg.api import CardType
for cid, c in db.all_cards().items():
    if c.cardType == CardType.SPECIAL_ENERGY.value:
        print(f"  id={cid} name={c.name}")

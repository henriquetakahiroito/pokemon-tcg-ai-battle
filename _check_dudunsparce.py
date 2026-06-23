import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db
from cg.api import all_card_data

db = get_db()

# Check Dudunsparce and Dunsparce full data including skills/abilities
for cid in [65, 66]:
    c = db.card(cid)
    print(f"id={cid} name={db.name(cid)} hp={c.hp} ex={c.ex}")
    for s in (c.skills or []):
        print(f"  ABILITY '{s.name}': {s.text}")
    for aid in (c.attacks or []):
        a = db.attack(aid)
        if a:
            print(f"  ATTACK {a.attackId} '{a.name}' dmg={a.damage} cost={a.cost} text='{a.text}'")

# Also check Legacy Energy full skill text
print("\n=== Legacy Energy ===")
c = db.card(12)
for s in (c.skills or []):
    print(f"  {s.name}: {s.text}")

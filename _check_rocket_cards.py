import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db
from cg.api import all_card_data

db = get_db()

# Check the Team Rocket cards and Secret Box
for cid in [1219, 1134, 1092]:
    c = db.card(cid)
    print(f"id={cid} name={db.name(cid)}")
    if c:
        from cg.api import CardType
        ct = CardType(c.cardType)
        print(f"  cardType={ct.name}")
        for s in (c.skills or []):
            print(f"  skill: {s.name}: {s.text}")
        for aid in (c.attacks or []):
            a = db.attack(aid)
            if a:
                print(f"  attack: {a.name} dmg={a.damage} text={a.text}")

# HP of all Hops Pokémon for prize trade analysis
print("\n=== Hops Pokémon HP ===")
for cid in [878, 879, 311, 304, 65, 66]:
    c = db.card(cid)
    if c:
        print(f"  id={cid} name={db.name(cid)} hp={c.hp} ex={c.ex} basic={c.basic}")

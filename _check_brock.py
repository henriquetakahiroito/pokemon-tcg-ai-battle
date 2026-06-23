import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db
db = get_db()

for cid in [1210, 1231, 1079, 1081]:
    c = db.card(cid)
    print(f"id={cid} name={db.name(cid)}")
    if c:
        from cg.api import CardType
        print(f"  type={CardType(c.cardType).name}")
        for s in (c.skills or []):
            print(f"  {s.name}: {s.text}")
        for aid in (c.attacks or []):
            a = db.attack(aid)
            if a:
                print(f"  ATTACK '{a.name}' dmg={a.damage}")

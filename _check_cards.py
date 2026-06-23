from agent.cards import get_db
db = get_db()
for cid in [878, 879, 311, 304, 65, 66]:
    print(f'id={cid} name={db.name(cid)}')
    for a in db.attacks_of(cid):
        print(f'  attack {a.attackId} "{a.name}" dmg={a.damage} cost={a.cost} text="{a.text[:90]}"')

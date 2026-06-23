import sys, json, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db
from cg.api import all_card_data
db = get_db()

# Check Mega Starmie ex and related cards
for cid in [1031, 1145, 1120, 1189, 1219]:
    c = db.card(cid)
    print(f"id={cid} name={db.name(cid)}")
    if c:
        from cg.api import CardType
        print(f"  type={CardType(c.cardType).name} ex={c.ex} hp={c.hp}")
        for s in (c.skills or []):
            print(f"  SKILL '{s.name}': {s.text[:150]}")
        for aid in (c.attacks or []):
            a = db.attack(aid)
            if a:
                print(f"  ATTACK '{a.name}' dmg={a.damage} cost={a.cost} text='{a.text[:120]}'")

# Check Crushing Hammer
c = db.card(1120)
print(f"\nid=1120 {db.name(1120)}")
if c:
    for s in (c.skills or []):
        print(f"  {s.name}: {s.text[:200]}")

# Isolate foo_foo's deck by looking at episodes where they're P0
foo_files_as_p0 = [
    r'C:\Users\Henrique\Downloads\80982156.json',      # foo_foo vs THIRD
    r'C:\Users\Henrique\Downloads\80986112 (1).json',  # foo_foo vs TeaParty
    r'C:\Users\Henrique\Downloads\81012267.json',
    r'C:\Users\Henrique\Downloads\81014817.json',
    r'C:\Users\Henrique\Downloads\81016102.json',
    r'C:\Users\Henrique\Downloads\81023738.json',
    r'C:\Users\Henrique\Downloads\81047221.json',
    r'C:\Users\Henrique\Downloads\81064026.json',
    r'C:\Users\Henrique\Downloads\81108279.json',
    r'C:\Users\Henrique\Downloads\81144029.json',
]

foofoo_cards = collections.Counter()
for path in foo_files_as_p0:
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            ep = json.load(f)
        names = [a['Name'] for a in ep['info']['Agents']]
        if 'foo_foo' not in names[0]:
            continue
        for step in ep['steps']:
            obs = step[0].get('observation', {}) or {}
            cur = obs.get('current') or {}
            if not cur:
                continue
            players = cur.get('players', [])
            if not players:
                continue
            p = players[0]
            for area in ('hand', 'active', 'bench', 'discard', 'prize'):
                for c in (p.get(area) or []):
                    if isinstance(c, dict) and c.get('id'):
                        foofoo_cards[c['id']] += 1
    except Exception as e:
        print(f"  Error {path}: {e}")

print("\n=== foo_foo deck (from P0 episodes) ===")
for cid, cnt in foofoo_cards.most_common(30):
    print(f"  id={cid} {db.name(cid)}: {cnt}")

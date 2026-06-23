import sys, json, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db
db = get_db()

files = [
    r'C:\Users\Henrique\Downloads\80982156.json',
    r'C:\Users\Henrique\Downloads\80984524 (1).json',
    r'C:\Users\Henrique\Downloads\80986112 (1).json',
    r'C:\Users\Henrique\Downloads\81012267.json',
    r'C:\Users\Henrique\Downloads\81014817.json',
    r'C:\Users\Henrique\Downloads\81016102.json',
    r'C:\Users\Henrique\Downloads\81023738.json',
    r'C:\Users\Henrique\Downloads\81047221.json',
    r'C:\Users\Henrique\Downloads\81058442.json',
    r'C:\Users\Henrique\Downloads\81060704.json',
    r'C:\Users\Henrique\Downloads\81063469.json',
    r'C:\Users\Henrique\Downloads\81064026.json',
    r'C:\Users\Henrique\Downloads\81084960.json',
    r'C:\Users\Henrique\Downloads\81108279.json',
    r'C:\Users\Henrique\Downloads\81141614.json',
    r'C:\Users\Henrique\Downloads\81144029.json',
    r'C:\Users\Henrique\Downloads\81160286.json',
]

all_results = []
p0_cards = collections.Counter()
p1_cards = collections.Counter()

for path in files:
    with open(path, encoding='utf-8', errors='replace') as f:
        ep = json.load(f)

    info = ep['info']
    names = [a['Name'] for a in info['Agents']]
    rewards = ep['rewards']
    eid = info['EpisodeId']
    won = [rewards[i] > 0 for i in range(2)]
    all_results.append((eid, names[0], names[1], won[0]))

    for step in ep['steps']:
        for pi in range(2):
            obs = step[pi].get('observation', {}) or {}
            cur = obs.get('current') or {}
            if not cur:
                continue
            players = cur.get('players', [])
            if not players or pi >= len(players):
                continue
            p = players[pi]
            counter = p0_cards if pi == 0 else p1_cards
            for area in ('hand', 'active', 'bench', 'discard', 'prize'):
                for c in (p.get(area) or []):
                    if isinstance(c, dict) and c.get('id'):
                        counter[c['id']] += 1

print("=== Episode results ===")
for eid, p0, p1, p0won in all_results:
    print(f"  {eid}: {p0} vs {p1} -> {'P0 WIN' if p0won else 'P1 WIN'}")

print("\n=== P0 top cards ===")
for cid, cnt in p0_cards.most_common(25):
    print(f"  id={cid} {db.name(cid)}: {cnt}")

print("\n=== P1 top cards ===")
for cid, cnt in p1_cards.most_common(25):
    print(f"  id={cid} {db.name(cid)}: {cnt}")

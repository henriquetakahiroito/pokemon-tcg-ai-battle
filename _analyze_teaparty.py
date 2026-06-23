import json, collections, os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db
db = get_db()

files = [
    r'C:\Users\Henrique\Downloads\80984524.json',
    r'C:\Users\Henrique\Downloads\80985240.json',
    r'C:\Users\Henrique\Downloads\80986112.json',
    r'C:\Users\Henrique\Downloads\80986116.json',
    r'C:\Users\Henrique\Downloads\80986873.json',
    r'C:\Users\Henrique\Downloads\80986874.json',
    r'C:\Users\Henrique\Downloads\80987604.json',
    r'C:\Users\Henrique\Downloads\80988338.json',
    r'C:\Users\Henrique\Downloads\80989071.json',
    r'C:\Users\Henrique\Downloads\80989633.json',
    r'C:\Users\Henrique\Downloads\80990376.json',
    r'C:\Users\Henrique\Downloads\81013548.json',
    r'C:\Users\Henrique\Downloads\81035633.json',
    r'C:\Users\Henrique\Downloads\81035625.json',
    r'C:\Users\Henrique\Downloads\81055362.json',
    r'C:\Users\Henrique\Downloads\81074382.json',
    r'C:\Users\Henrique\Downloads\81091518.json',
    r'C:\Users\Henrique\Downloads\81112807.json',
    r'C:\Users\Henrique\Downloads\81142775.json',
]

card_counter = collections.Counter()
opp_counter = collections.Counter()
results = []

for path in files:
    with open(path, encoding='utf-8', errors='replace') as f:
        ep = json.load(f)

    info = ep['info']
    names = [a['Name'] for a in info['Agents']]
    rewards = ep['rewards']

    tea_idx = next((i for i, n in enumerate(names) if 'Tea Party' in n or 'Debauchery' in n), 0)
    opp_idx = 1 - tea_idx
    won = rewards[tea_idx] > 0
    opp_name = names[opp_idx]
    opp_counter[opp_name] += 1

    eid = info['EpisodeId']
    results.append((eid, names[tea_idx], opp_name, won))

    for step in ep['steps']:
        obs = step[tea_idx].get('observation', {})
        if not obs:
            continue
        cur = obs.get('current', {})
        if not cur:
            continue
        players = cur.get('players', [])
        if not players or tea_idx >= len(players):
            continue
        p = players[tea_idx]
        for area_key in ('hand', 'active', 'bench', 'discard', 'prize'):
            for c in (p.get(area_key) or []):
                if isinstance(c, dict) and c.get('id'):
                    card_counter[c['id']] += 1

print("=== Episode results ===")
for eid, me, opp, won in results:
    outcome = "WIN" if won else "LOSS"
    print(f"  {eid}: {me} vs {opp} -> {outcome}")

print("\n=== Top cards seen in Tea Party player state (likely their deck) ===")
for cid, cnt in card_counter.most_common(35):
    print(f"  id={cid} name={db.name(cid)} count={cnt}")

print("\n=== Opponents faced ===")
for name, cnt in opp_counter.most_common():
    print(f"  {name}: {cnt} games")

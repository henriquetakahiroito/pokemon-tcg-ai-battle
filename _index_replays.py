"""Token saver: parse each replay ONCE into a tiny CSV. Re-run anytime — it only parses
NEW episodes and appends. Then analysis reads the small CSV instead of re-parsing megabytes.

Usage:  python _index_replays.py            # scans Downloads, updates _replay_index.csv
Columns: episode, p0, p1, my_seat(if H.Ito), my_deck, opp_archetype, winner, result_me"""
import sys, json, glob, os, csv; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from collections import Counter
from agent.cards import get_db
db=get_db(); DL=r'C:\Users\Henrique\Downloads'; IDX='_replay_index.csv'
def arch(deck):
    c=Counter(deck)
    if c.get(1031,0): return 'MegaStarmie'
    if c.get(121,0) or c.get(119,0): return 'Dragapult'
    if c.get(678,0) or c.get(677,0): return 'MegaLucario'
    if c.get(943,0): return 'Walrein'
    if c.get(431,0) or c.get(401,0): return 'TR-Spidops'
    if c.get(743,0) or c.get(742,0): return 'Alakazam'
    if c.get(272,0) and c.get(66,0): return 'Hops-Clefairy'
    if c.get(66,0) and c.get(879,0): return 'Hops'
    if c.get(163,0) or c.get(144,0): return 'Slowking'
    if c.get(345,0): return 'Crustle'
    if c.get(164,0): return 'Comfey'
    return 'other:'+str([x for x,_ in c.most_common(4) if x>20][:3])
def deck_of(d,i):
    for step in d['steps']:
        a=step[i].get('action')
        if isinstance(a,list) and len(a)==60: return a
    return None
seen=set()
if os.path.exists(IDX):
    for r in csv.DictReader(open(IDX,encoding='utf-8')): seen.add(r['episode'])
rows=[]; nfiles=0
for fp in glob.glob(os.path.join(DL,'*.json')):
    if '-0' in fp or '-1' in fp: continue
    eid=os.path.splitext(os.path.basename(fp))[0].split(' ')[0]
    if eid in seen: continue
    try: d=json.load(open(fp,encoding='utf-8'))
    except Exception: continue
    if not isinstance(d,dict) or 'steps' not in d: continue
    names=d.get('info',{}).get('TeamNames') or ['?','?']; rew=d.get('rewards') or []
    real_eid=str(d.get('info',{}).get('EpisodeId',eid))
    if real_eid in seen: continue
    seen.add(real_eid); nfiles+=1
    d0=deck_of(d,0); d1=deck_of(d,1)
    me = names.index('H.Ito') if 'H.Ito' in names else -1
    winner = names[0] if (rew and rew[0]==1) else (names[1] if (rew and len(rew)>1 and rew[1]==1) else 'draw')
    rows.append({'episode':real_eid,'p0':names[0],'p1':names[1] if len(names)>1 else '?',
                 'my_seat':me,'my_deck':arch(d0) if (me==0 and d0) else (arch(d1) if (me==1 and d1) else ''),
                 'opp_archetype':(arch(d1) if me==0 and d1 else arch(d0) if me==1 and d0 else ''),
                 'winner':winner,'result_me':(rew[me] if (me>=0 and me<len(rew)) else '')})
new=not os.path.exists(IDX)
with open(IDX,'a',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['episode','p0','p1','my_seat','my_deck','opp_archetype','winner','result_me'])
    if new: w.writeheader()
    w.writerows(rows)
print(f'indexed {nfiles} new episodes -> {IDX} (total cached: {len(seen)})')
print('Now analysis reads the CSV, not the JSONs. e.g. your win rate by matchup:')
import collections
wl=collections.defaultdict(lambda:[0,0])
for r in csv.DictReader(open(IDX,encoding='utf-8')):
    if r['my_deck'] and r['result_me'] in ('1','-1'):
        wl[(r['my_deck'],r['opp_archetype'])][0 if r['result_me']=='1' else 1]+=1
for (md,oa),(w_,l_) in sorted(wl.items()):
    if w_+l_>=2: print(f'  {md:<14} vs {oa:<14} {w_}-{l_}')

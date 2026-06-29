"""Audit H.Ito's REAL ladder Archaludon card-PLAY decisions vs the user's complaint:
'why scoop + retreat, why not Ultra Ball, why Black Belt not Lillie'.

For H.Ito's seat across all Archaludon games, resolve each chosen PLAY option to its card id
(hand[index]) and tally the complaint cards; also count how often Ultra Ball was OFFERED but
NOT played (missed digs) and how often RETREAT was taken. Split WIN vs LOSS.
"""
import sys, json, glob, os, csv, collections
sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
DL=r'C:\Users\Henrique\Downloads\JSONMatchreplays'
PLAY,RETREAT=7,12
CARDS={1093:'ScoopUpCyclone',1121:'UltraBall',1211:'BlackBelt',1227:'Lillie',
       1192:'Carmine',1182:'Boss',1097:'NightStretcher',1122:'Pokegear'}

rows=[r for r in csv.DictReader(open('_replay_index.csv',encoding='utf-8'))]
games={r['episode']:r for r in rows if r['my_deck'] in ('Archaludon',) or r['my_deck'].startswith('other:[169')}
games={k:v for k,v in games.items() if v['result_me'] in ('1','-1')}

played={'1':collections.Counter(),'-1':collections.Counter()}
offered_ub={'1':[0,0],'-1':[0,0]}   # [offered, played]
retreats={'1':0,'-1':0}; ng={'1':0,'-1':0}

for fp in glob.glob(os.path.join(DL,'*.json')):
    eid=os.path.splitext(os.path.basename(fp))[0].split(' ')[0]
    g=games.get(eid)
    if not g: continue
    try: d=json.load(open(fp,encoding='utf-8'))
    except Exception: continue
    g=games.get(str(d.get('info',{}).get('EpisodeId',eid))) or g
    res=g['result_me']; seat=int(g['my_seat']); ng[res]+=1
    for step in d['steps']:
        cell=step[seat]; act=cell.get('action'); obs=cell.get('observation')
        if not isinstance(act,list) or not isinstance(obs,dict): continue
        sel=obs.get('select');  st=obs.get('current') or {}
        if not isinstance(sel,dict): continue
        opts=sel.get('option') or []
        players=st.get('players') or []
        hand=players[seat].get('hand') if seat<len(players) else None
        def cid_of(o):
            i=o.get('index')
            if hand and isinstance(i,int) and 0<=i<len(hand): return (hand[i] or {}).get('id')
            return None
        # offered Ultra Ball this decision?
        ub_off=any(o.get('type')==PLAY and cid_of(o)==1121 for o in opts)
        ub_play=False
        for i in act:
            if not isinstance(i,int) or i>=len(opts): continue
            o=opts[i]; t=o.get('type')
            if t==PLAY:
                c=cid_of(o)
                if c in CARDS: played[res][c]+=1
                if c==1121: ub_play=True
            elif t==RETREAT:
                retreats[res]+=1
        if ub_off:
            offered_ub[res][0]+=1
            if ub_play: offered_ub[res][1]+=1

for res,label in (('1','WINS'),('-1','LOSSES')):
    n=ng[res]
    print(f"\n=== {label} ({n} games) ===")
    pc=played[res]
    for cid,nm in CARDS.items():
        print(f"   {nm:<16} played {pc[cid]}")
    off,pl=offered_ub[res]
    print(f"   Ultra Ball OFFERED {off}, PLAYED {pl}  -> missed digs {off-pl} ({(off-pl)/off:.0%} ignored)" if off else "   Ultra Ball never offered")
    print(f"   RETREATS taken: {retreats[res]}  ({retreats[res]/n:.1f}/game)")

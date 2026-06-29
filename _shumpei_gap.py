"""Why are we still losing? Compare ShumpeiNomura (top Archaludon, 72%) to H.Ito's Archaludon
LOSSES at the action level, and show Shumpei's win progression over his games.
"""
import sys, json, glob, os, csv, collections
sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
DL=r'C:\Users\Henrique\Downloads\JSONMatchreplays'
PLAY,RETREAT,ATTACK,EVOLVE,ABILITY,ATTACH=7,12,13,9,10,8
_MD,_IRON,_ARCH_EX,_DURA=253,225,190,169

rows=[r for r in csv.DictReader(open('_replay_index.csv',encoding='utf-8'))]
def file_for(eid):
    for c in (eid,):
        p=os.path.join(DL,c+'.json')
        if os.path.exists(p): return p
    return None

def seat_of(r,name): return 0 if r['p0']==name else (1 if r['p1']==name else -1)

# ---- Shumpei progression (episode-id order ~ chronological) ----
sg=[(r['episode'],r,seat_of(r,'ShumpeiNomura')) for r in rows
    if 'ShumpeiNomura' in (r['p0'],r['p1'])]
sg.sort(key=lambda x:int(x[0]))
cw=ct=0; marks=[]
for eid,r,seat in sg:
    ct+=1; cw += 1 if r['winner']=='ShumpeiNomura' else 0
    if ct in (10,20,30,40,50,60,70,82): marks.append((ct,cw/ct))
print("ShumpeiNomura progression (cumulative winrate by game #):")
print("   " + "  ".join(f"g{n}:{wr:.0%}" for n,wr in marks))

def profile(episode_seat_list, label):
    atk=collections.Counter(); ev=collections.Counter(); n=0; md_games=0; ko_dealt=0; turns=[]
    for eid,seat in episode_seat_list:
        fp=file_for(eid)
        if not fp: continue
        try: d=json.load(open(fp,encoding='utf-8'))
        except Exception: continue
        n+=1; got_md=False
        for ti,step in enumerate(d['steps']):
            cell=step[seat]; act=cell.get('action'); obs=cell.get('observation')
            if not isinstance(act,list) or not isinstance(obs,dict): continue
            sel=obs.get('select'); st=obs.get('current') or {}
            if not isinstance(sel,dict): continue
            opts=sel.get('option') or []
            players=st.get('players') or []
            hand=players[seat].get('hand') if seat<len(players) else None
            for i in act:
                if not isinstance(i,int) or i>=len(opts): continue
                o=opts[i]; t=o.get('type')
                if t==ATTACK:
                    aid=o.get('attackId'); atk[aid]+=1
                    if aid in (_MD,_IRON) and not got_md: got_md=True; turns.append(ti)
                elif t==EVOLVE:
                    idx=o.get('index')
                    cid=(hand[idx]or{}).get('id') if (hand and isinstance(idx,int) and idx<len(hand)) else None
                    ev['archex' if cid==_ARCH_EX else 'other']+=1
        if got_md: md_games+=1
    tot=sum(atk.values()) or 1
    md=atk[_MD]+atk[_IRON]; ham=atk[223]+atk[224]
    print(f"\n=== {label}  ({n} games) ===")
    print(f"  Metal Defender/Iron: {md} ({md/tot:.0%})   raw-Duraludon hammer: {ham} ({ham/tot:.0%})")
    print(f"  evolve->Archaludon ex: {ev['archex']}   other: {ev['other']}")
    print(f"  reached Metal Defender: {md_games}/{n} ({md_games/n:.0%})" + (f"   median first-MD step {sorted(turns)[len(turns)//2]}" if turns else "  NEVER"))
    print(f"  top attacks: " + ", ".join(f"{a}:{c}" for a,c in atk.most_common(5)))

# Shumpei is Archaludon — profile ALL his games on his seat
shump=[(eid,seat) for eid,r,seat in sg if seat>=0]
profile(shump, "SHUMPEI (Archaludon, all games)")

# Our Archaludon LOSSES
ours_loss=[(r['episode'], int(r['my_seat'])) for r in rows
           if (r['my_deck']=='Archaludon' or r['my_deck'].startswith('other:[169'))
           and r['result_me']=='-1']
profile(ours_loss, "H.Ito Archaludon LOSSES")
ours_win=[(r['episode'], int(r['my_seat'])) for r in rows
          if (r['my_deck']=='Archaludon' or r['my_deck'].startswith('other:[169'))
          and r['result_me']=='1']
profile(ours_win, "H.Ito Archaludon WINS")

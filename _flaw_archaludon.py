"""Decode H.Ito's REAL ladder Archaludon games (from Downloads replays) and find the
systematic flaw: compare what the agent DID in losses vs wins.

Per game, on H.Ito's seat, tally chosen options:
  - evolve into Archaludon ex(190)
  - attack ids: Metal Defender(253) vs raw Duraludon hammer(223/224) vs other
  - turn index of FIRST Metal Defender (tempo: how fast the engine comes online)
Aggregate split by game result (WIN vs LOSS).
"""
import sys, json, glob, os, collections
sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import csv
from cg.api import OptionType
EV_T, AT_T = OptionType.EVOLVE.value, OptionType.ATTACK.value
DL=r'C:\Users\Henrique\Downloads\JSONMatchreplays'
_ARCH_EX, _DURA = 190, 169
_MD, _IRON = 253, 225

# load my Archaludon episodes + results from the index
rows=[r for r in csv.DictReader(open('_replay_index.csv',encoding='utf-8'))]
games={r['episode']:r for r in rows if r['my_deck'].startswith('other:[169') and r['result_me'] in ('1','-1')}

def opt_cid(o, state):
    # resolve the card id an option refers to (best-effort, mirrors policy._card_id_from_option)
    for k in ('cardId','id'):
        if k in o and isinstance(o[k],int): return o[k]
    idx=o.get('cardIndex'); zone=o.get('zone')
    return None

agg={'1':collections.Counter(),'-1':collections.Counter()}
attacks={'1':collections.Counter(),'-1':collections.Counter()}
first_md={'1':[], '-1':[]}
ngame={'1':0,'-1':0}

for fp in glob.glob(os.path.join(DL,'*.json')):
    eid=os.path.splitext(os.path.basename(fp))[0].split(' ')[0]
    g=games.get(eid)
    if not g: continue
    try: d=json.load(open(fp,encoding='utf-8'))
    except Exception: continue
    reid=str(d.get('info',{}).get('EpisodeId',eid))
    g=games.get(reid) or g
    res=g['result_me']; seat=int(g['my_seat'])
    ngame[res]+=1
    md_turn=None
    for ti,step in enumerate(d['steps']):
        cell=step[seat]
        act=cell.get('action'); obs=cell.get('observation')
        if not isinstance(act,list) or not isinstance(obs,dict): continue
        sel=obs.get('select')
        if not isinstance(sel,dict): continue
        opts=sel.get('option') or []
        state=obs.get('current') or {}
        for i in act:
            if not isinstance(i,int) or i>=len(opts): continue
            o=opts[i]; t=o.get('type')
            if t==EV_T:
                cid=opt_cid(o,state)
                agg[res]['evolve_archex' if cid==_ARCH_EX else 'evolve_other']+=1
            elif t==AT_T:
                aid=o.get('attackId')
                attacks[res][aid]+=1
                if aid in (_MD,_IRON) and md_turn is None:
                    md_turn=ti
    if md_turn is not None: first_md[res].append(md_turn)

def show(res,label):
    n=ngame[res]; at=attacks[res]; tot=sum(at.values()) or 1
    md=at[_MD]+at[_IRON]; ham=at[223]+at[224]
    fm=first_md[res]
    print(f"\n=== {label}  ({n} games) ===")
    print(f"  evolve into Archaludon ex: {agg[res]['evolve_archex']}   other-evolve: {agg[res]['evolve_other']}")
    print(f"  attacks: Metal Defender/Iron {md} ({md/tot:.0%})   raw-Duraludon hammer {ham} ({ham/tot:.0%})   total {tot}")
    print(f"  games that reached Metal Defender: {len(fm)}/{n}" + (f"   median FIRST-MD step ~{sorted(fm)[len(fm)//2]}" if fm else "  (NEVER online!)"))
    top=at.most_common(6)
    print(f"  attack mix: " + ", ".join(f"{a}:{c}" for a,c in top))

show('1',"WINS")
show('-1',"LOSSES")
print("\nNOTE: attackId 223/224=Duraludon hammer, 253=Metal Defender, 225=Iron Blaster, 1006=Gobble, 1007=Huge Bite")

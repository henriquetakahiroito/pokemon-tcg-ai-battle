"""Does the behavior-cloned scorer pilot Walrein better than the hand heuristic?
numpy inference (submission-safe) of the BC net, used as a greedy policy, vs the heuristic."""
import sys, random; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from cg.api import OptionType, CardType
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose
from agent.base import read_deck
from agent.cards import get_db
db=get_db()
W=np.load('_bc_weights.npz'); mu=W['mu']; sd=W['sd']
def mlp(x):
    h=np.maximum(0, x@W['W0'].T+W['b0']); h=np.maximum(0,h@W['W1'].T+W['b1']); return (h@W['W2'].T+W['b2']).ravel()
OPT_TYPES=[t.value for t in OptionType]; CARD_TYPES=[c.value for c in CardType]
def card_id(o,state):
    area=o.get('area'); idx=o.get('index'); pi=o.get('playerIndex')
    if area is None or idx is None: return None
    try: pl=state['players'][pi if pi is not None else state['yourIndex']]
    except Exception: return None
    key={1:'deck',2:'hand',3:'discard',5:'bench',12:'looking'}.get(area)
    if area==4:
        a=pl.get('active') or []; return a[0].get('id') if a and a[0] else None
    if key:
        arr=pl.get(key) or []
        if 0<=idx<len(arr) and arr[idx]: return arr[idx].get('id')
    return None
def feat(o,state,ctx):
    me=state['yourIndex']; mp=state['players'][me]; op=state['players'][1-me]; f=[]
    t=o.get('type'); f+=[1.0 if t==tv else 0.0 for tv in OPT_TYPES]
    cid=card_id(o,state); ct=db.card_type(cid).value if (cid and db.card_type(cid)) else -1
    f+=[1.0 if ct==cv else 0.0 for cv in CARD_TYPES]
    c=db.card(cid) if cid else None
    f.append(1.0 if (c and getattr(c,'basic',False)) else 0.0); f.append(1.0 if (c and getattr(c,'ex',False)) else 0.0)
    aid=o.get('attackId'); ai=db.attack(aid) if aid is not None else None
    f.append((ai.damage/100.0) if ai else 0.0); f.append((ai.cost/3.0) if ai else 0.0)
    f.append(len(mp.get('prize') or [])/6.0); f.append(len(op.get('prize') or [])/6.0)
    f.append(min(mp.get('handCount',len(mp.get('hand') or [])),12)/12.0)
    f.append(len(mp.get('bench') or [])/5.0); f.append(len(op.get('bench') or [])/5.0); f.append(min(ctx,48)/48.0)
    return f
def bc_choose(obs,rng):
    sel=obs.get('select')
    if sel is None: return None
    opts=sel.get('option') or []; n=len(opts)
    lo,hi=sel['minCount'],min(sel['maxCount'],n)
    if hi<=0: return []
    k=lo if lo>0 else 1
    if n<2 or hi!=lo and not (lo<=1<=hi): return choose(obs,rng=rng)  # multi-select -> heuristic
    if lo>1: return choose(obs,rng=rng)
    state=obs['current']; ctx=sel.get('context',0)
    Xo=np.array([feat(o,state,ctx) for o in opts],dtype=np.float32)
    s=mlp((Xo-mu)/sd)
    return sorted(np.argsort(-s)[:max(1,k)].tolist())
WAL=read_deck('deck_cand_walrein.csv'); WALREIN,DUD=943,66
FRIG=[a.attackId for a in db.attacks_of(WALREIN) if 'Frigid' in a.name][0]
MEG=[a.attackId for a in db.attacks_of(WALREIN) if 'Megaton' in a.name][0]; HAM={1081,1120}
def has(p,cid):
    a=p.get('active') or []
    if a and a[0] and a[0].get('id')==cid: return True
    return any(b and b.get('id')==cid for b in (p.get('bench') or []))
def pol_bc(deck,seed):
    r=random.Random(seed)
    def f(obs): return list(deck) if obs.get('select') is None else bc_choose(obs,r)
    return f
def pol_h(deck,seed):
    r=random.Random(seed)
    def f(obs): return list(deck) if obs.get('select') is None else choose(obs,rng=r)
    return f
def run(label,herofac,opp,n=50):
    a=herofac(WAL,1); b=pol_h(opp,2); wins=games=rw=fr=mg=du=hm=0
    for g in range(n):
        a0=(g%2==0); me=0 if a0 else 1
        obs,_=battle_start(WAL if a0 else opp, opp if a0 else WAL)
        if obs is None: continue
        agents=(a,b) if a0 else (b,a); steps=0; rW=fF=mF=dD=hH=False
        try:
            while True:
                cur=obs.get('current')
                if cur and cur.get('result',-1)!=-1:
                    if cur['result']==me: wins+=1
                    break
                sel=obs.get('select')
                if sel is None: break
                who=cur['yourIndex']
                if who==me:
                    if has(cur['players'][me],WALREIN): rW=True
                    ch=agents[who](obs)
                    for idx in (ch or []):
                        opts=sel.get('option') or []
                        if 0<=idx<len(opts):
                            o=opts[idx]; t=o.get('type')
                            if t==OptionType.ATTACK.value:
                                if o.get('attackId')==FRIG: fF=True
                                elif o.get('attackId')==MEG: mF=True
                            elif t==OptionType.PLAY.value and card_id(o,cur) in HAM: hH=True
                            elif t==OptionType.ABILITY.value:
                                ipa=o.get('inPlayArea'); ipi=o.get('inPlayIndex'); mp=cur['players'][me]
                                tg=(mp.get('active') or [None])[0] if ipa==4 else (mp.get('bench') or [])[ipi] if (ipa==5 and ipi is not None and ipi<len(mp.get('bench') or [])) else None
                                if tg and tg.get('id')==DUD: dD=True
                    obs=battle_select(ch)
                else: obs=battle_select(agents[who](obs))
                steps+=1
                if steps>30000: break
        finally: battle_finish()
        games+=1; rw+=rW; fr+=fF; mg+=mF; du+=dD; hm+=hH
    print(f'  [{label}] vs Dragapult: {wins}/{games}={wins/games:.0%} | reachWalrein {rw}/{games}, Frigid {fr}, Megaton {mg}, Dudun {du}, Hammer {hm}')

print('Walrein piloting: heuristic vs behavior-cloned policy (50 games vs Dragapult)')
run('heuristic', pol_h, read_deck('deck_cand_dragapult_real.csv'))
run('BC-clone ', pol_bc, read_deck('deck_cand_dragapult_real.csv'))

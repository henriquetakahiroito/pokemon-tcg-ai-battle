"""Does BC v3 (board features, winners-trained) use Dudunsparce's ability in v2c,
where the hand-heuristic was 0/50?"""
import sys, random, pickle; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from cg.api import OptionType, CardType
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose
from agent.base import read_deck
from agent.cards import get_db
db=get_db(); W=np.load('_bc_weights3.npz'); vocab=pickle.load(open('_bc_vocab3.pkl','rb'))
mu=W['mu']; sd=W['sd']; emb=W['emb']
OPT=[t.value for t in OptionType]; CT=[c.value for c in CardType]
from cg.api import CardType as CTp
def is_se(cid): c=db.card(cid); return bool(c and c.cardType==CTp.SPECIAL_ENERGY.value)
def cidf(o,s):
    area=o.get('area'); idx=o.get('index'); pi=o.get('playerIndex')
    if area is None or idx is None: return None
    try: pl=s['players'][pi if pi is not None else s['yourIndex']]
    except Exception: return None
    k={1:'deck',2:'hand',3:'discard',5:'bench',12:'looking'}.get(area)
    if area==4:
        a=pl.get('active') or []; return a[0].get('id') if a and a[0] else None
    if k:
        arr=pl.get(k) or []
        if 0<=idx<len(arr) and arr[idx]: return arr[idx].get('id')
    return None
def board(state):
    me=state['yourIndex']; mp=state['players'][me]; op=state['players'][1-me]
    def act(p):
        a=p.get('active') or []; return a[0] if a and a[0] else None
    def en(pk): return len(pk.get('energies') or []) if pk else 0
    def hp(pk):
        if not pk: return 0.0
        h=pk.get('hp',0) or 0; m=pk.get('maxHp',h) or h or 1; return h/m
    def eid(e): return e.get('id') if isinstance(e,dict) else e
    ma=act(mp); oa=act(op)
    osp=sum(1 for pk in [oa]+list(op.get('bench') or []) if pk for e in (pk.get('energies') or []) if is_se(eid(e)))
    return [hp(ma),en(ma)/4.0,hp(oa),en(oa)/4.0,
            1.0 if (oa and db.card(oa.get('id')) and getattr(db.card(oa.get('id')),'ex',False)) else 0.0,
            osp/4.0, sum(en(x) for x in [ma]+list(mp.get('bench') or []) if x)/8.0,
            min(len(mp.get('deck') or []) or mp.get('deckCount',0),40)/40.0]
def scal(o,s,ctx,bd):
    me=s['yourIndex']; mp=s['players'][me]; op=s['players'][1-me]; f=[]
    t=o.get('type'); f+=[1.0 if t==tv else 0.0 for tv in OPT]
    cid=cidf(o,s); ct=db.card_type(cid).value if (cid and db.card_type(cid)) else -1
    f+=[1.0 if ct==cv else 0.0 for cv in CT]
    c=db.card(cid) if cid else None
    f.append(1.0 if (c and getattr(c,'basic',False)) else 0.0); f.append(1.0 if (c and getattr(c,'ex',False)) else 0.0)
    aid=o.get('attackId'); ai=db.attack(aid) if aid is not None else None
    f.append((ai.damage/100.0) if ai else 0.0); f.append((ai.cost/3.0) if ai else 0.0)
    f.append(len(mp.get('prize') or [])/6.0); f.append(len(op.get('prize') or [])/6.0)
    f.append(min(mp.get('handCount',len(mp.get('hand') or [])),12)/12.0)
    f.append(len(mp.get('bench') or [])/5.0); f.append(len(op.get('bench') or [])/5.0); f.append(min(ctx,48)/48.0)
    return f+bd
def mlp(xs,ci):
    x=np.concatenate([xs,emb[ci]],1)
    h=np.maximum(0,x@W['W0'].T+W['b0']); h=np.maximum(0,h@W['W1'].T+W['b1']); return (h@W['W2'].T+W['b2']).ravel()
def bc(obs,r):
    sel=obs.get('select')
    if sel is None: return None
    opts=sel.get('option') or []; n=len(opts); lo,hi=sel['minCount'],min(sel['maxCount'],n)
    if hi<=0: return []
    if n<2 or lo>1 or not (lo<=1<=hi): return choose(obs,rng=r)
    s=obs['current']; ctx=sel.get('context',0); bd=board(s)
    xs=np.array([scal(o,s,ctx,bd) for o in opts],np.float32); xs=(xs-mu)/sd
    ci=np.array([vocab.get(cidf(o,s),0) for o in opts])
    return sorted(np.argsort(-mlp(xs,ci))[:1].tolist())
V2C=read_deck('deck_cand_hops_hybrid_v2c.csv'); DRAG=read_deck('deck_cand_dragapult_real.csv'); DUD=66
def has(p,c):
    a=p.get('active') or []
    if a and a[0] and a[0].get('id')==c: return True
    return any(b and b.get('id')==c for b in (p.get('bench') or []))
def bpol(d,s):
    r=random.Random(s); return lambda o: list(d) if o.get('select') is None else bc(o,r)
def hpol(d,s):
    r=random.Random(s); return lambda o: list(d) if o.get('select') is None else choose(o,rng=r)
a=bpol(V2C,1); b=hpol(DRAG,2); wins=reachD=dudAb=0
for g in range(40):
    obs,_=battle_start(V2C,DRAG)
    if obs is None: continue
    ags=(a,b); steps=0; rD=dA=False
    try:
        while True:
            cur=obs.get('current')
            if cur and cur.get('result',-1)!=-1:
                if cur['result']==0: wins+=1
                break
            sel=obs.get('select')
            if sel is None: break
            who=cur['yourIndex']
            if who==0:
                if has(cur['players'][0],DUD): rD=True
                ch=ags[0](obs)
                for idx in (ch or []):
                    opts=sel.get('option') or []
                    if 0<=idx<len(opts):
                        o=opts[idx]
                        if o.get('type')==OptionType.ABILITY.value:
                            ipa=o.get('inPlayArea'); ipi=o.get('inPlayIndex'); mp=cur['players'][0]
                            tg=(mp.get('active') or [None])[0] if ipa==4 else (mp.get('bench') or [])[ipi] if (ipa==5 and ipi is not None and ipi<len(mp.get('bench') or [])) else None
                            if tg and tg.get('id')==DUD: dA=True
                obs=battle_select(ch)
            else: obs=battle_select(ags[who](obs))
            steps+=1
            if steps>30000: break
    finally: battle_finish()
    reachD+=rD; dudAb+=dA
print(f'BC v3 piloting v2c vs Dragapult (40 games): wins {wins}/40 | reach Dudunsparce {reachD}/40, USED Dudunsparce ability {dudAb}/40')
print('(hand-heuristic used Dudunsparce ability 0/50)')

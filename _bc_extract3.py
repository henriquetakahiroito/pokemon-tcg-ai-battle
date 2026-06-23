"""BC v3: richer BOARD-CONTEXT features + train only on WINNERS' decisions (quality filter
≈ good decision-making). Adds the info the model needs to learn WHEN to act (opp energy,
my hand/deck size, can-I-attack), so it can learn ability/disruption timing the heuristic skips."""
import sys, json, glob, os, pickle; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from cg.api import OptionType, CardType
from agent.cards import get_db
db=get_db(); DL=r'C:\Users\Henrique\Downloads'
OPT=[t.value for t in OptionType]; CT=[c.value for c in CardType]
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
def is_special_energy(cid):
    c=db.card(cid);
    from cg.api import CardType as CTp
    return bool(c and c.cardType==CTp.SPECIAL_ENERGY.value)
def board(state):
    me=state['yourIndex']; mp=state['players'][me]; op=state['players'][1-me]
    def act(p):
        a=p.get('active') or []; return a[0] if a and a[0] else None
    ma=act(mp); oa=act(op)
    def en(pk): return len(pk.get('energies') or []) if pk else 0
    def hp(pk):
        if not pk: return 0.0
        h=pk.get('hp',0) or 0; m=pk.get('maxHp',h) or h or 1; return h/m
    def eid(e): return e.get('id') if isinstance(e,dict) else e
    def opp_special(p):
        n=0
        for pk in ([act(p)]+list(p.get('bench') or [])):
            if pk:
                for e in (pk.get('energies') or []):
                    if is_special_energy(eid(e)): n+=1
        return n
    return [
        hp(ma), en(ma)/4.0, hp(oa), en(oa)/4.0,
        1.0 if (oa and db.card(oa.get('id')) and getattr(db.card(oa.get('id')),'ex',False)) else 0.0,
        opp_special(op)/4.0,
        sum(en(x) for x in ([ma]+list(mp.get('bench') or [])) if x)/8.0,   # my total energy
        min(len(mp.get('deck') or []) or mp.get('deckCount',0),40)/40.0,    # my deck size (deckout)
    ]
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
files=[f for f in glob.glob(os.path.join(DL,'*.json')) if '-0' not in f and '-1' not in f]
X=[]; C=[]; Y=[]; G=[]; vocab={None:0}; gid=0; ng=0
for fp in files:
    try: d=json.load(open(fp,encoding='utf-8'))
    except Exception: continue
    if not isinstance(d,dict) or 'steps' not in d: continue
    rew=d.get('rewards') or []
    ng+=1
    for step in d['steps']:
        for seat,ag in enumerate(step):
            if seat>=len(rew) or rew[seat]!=1: continue   # WINNERS ONLY
            obs=ag.get('observation') or {}; sel=obs.get('select'); cur=obs.get('current'); act=ag.get('action')
            if not sel or not cur or cur.get('yourIndex')!=seat: continue
            opts=sel.get('option') or []
            if len(opts)<2 or not isinstance(act,list) or len(act)!=1: continue
            ch=act[0]
            if not (0<=ch<len(opts)): continue
            ctx=sel.get('context',0); bd=board(cur)
            for i,o in enumerate(opts):
                cid=cidf(o,cur)
                if cid not in vocab: vocab[cid]=len(vocab)
                X.append(scal(o,cur,ctx,bd)); C.append(vocab[cid]); Y.append(1.0 if i==ch else 0.0); G.append(gid)
            gid+=1
X=np.array(X,np.float32); C=np.array(C,np.int64); Y=np.array(Y,np.float32); G=np.array(G,np.int64)
print(f'{ng} games (winners only) -> {gid} decisions, {len(X)} rows, {X.shape[1]} feats, vocab={len(vocab)}')
np.savez_compressed('_bc_data3.npz', X=X, C=C, Y=Y, G=G, vocab_size=len(vocab))
pickle.dump(vocab, open('_bc_vocab3.pkl','wb'))
print('saved _bc_data3.npz + _bc_vocab3.pkl')

"""Behavioral-cloning step 1: parse all replay JSONs into (option-features, chosen) rows.
Each single-select decision -> one feature row per legal option, label=1 if the player chose it.
We learn an option-SCORER (learning-to-rank via per-option binary classification)."""
import sys, json, glob, os; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from cg.api import OptionType, CardType
from agent.cards import get_db
db=get_db()
DL=r'C:\Users\Henrique\Downloads'
OPT_TYPES=[t.value for t in OptionType]
CARD_TYPES=[c.value for c in CardType]

def card_id(o, state):
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

def feat(o, state, ctx):
    me=state['yourIndex']; mp=state['players'][me]; op=state['players'][1-me]
    f=[]
    t=o.get('type')
    f+= [1.0 if t==tv else 0.0 for tv in OPT_TYPES]              # option type one-hot
    cid=card_id(o,state)
    ct=db.card_type(cid).value if (cid and db.card_type(cid)) else -1
    f+= [1.0 if ct==cv else 0.0 for cv in CARD_TYPES]           # card type one-hot
    # card props
    c=db.card(cid) if cid else None
    f.append(1.0 if (c and getattr(c,'basic',False)) else 0.0)
    f.append(1.0 if (c and getattr(c,'ex',False)) else 0.0)
    # attack props
    aid=o.get('attackId'); ai=db.attack(aid) if aid is not None else None
    f.append((ai.damage/100.0) if ai else 0.0)
    f.append((ai.cost/3.0) if ai else 0.0)
    # context (shared across options in a decision but gives the net board awareness)
    f.append(len(mp.get('prize') or [])/6.0)
    f.append(len(op.get('prize') or [])/6.0)
    f.append(min(mp.get('handCount',len(mp.get('hand') or [])),12)/12.0)
    f.append(len(mp.get('bench') or [])/5.0)
    f.append(len(op.get('bench') or [])/5.0)
    f.append(min(ctx,48)/48.0)                                  # select context id
    return f

X=[]; Y=[]; G=[]  # features, label, decision-group id
gid=0; files=[f for f in glob.glob(os.path.join(DL,'*.json')) if '-0' not in f and '-1' not in f]
nfiles=0
for fp in files:
    try: d=json.load(open(fp,encoding='utf-8'))
    except Exception: continue
    if not isinstance(d,dict) or 'steps' not in d: continue
    nfiles+=1
    for step in d['steps']:
        for seat,ag in enumerate(step):
            obs=ag.get('observation') or {}; sel=obs.get('select'); cur=obs.get('current')
            act=ag.get('action')
            if not sel or not cur or cur.get('yourIndex')!=seat: continue
            opts=sel.get('option') or []
            if len(opts)<2: continue                # skip forced single-option
            if not isinstance(act,list) or len(act)!=1: continue   # single-select only
            chosen=act[0]
            if not (0<=chosen<len(opts)): continue
            ctx=sel.get('context',0)
            for i,o in enumerate(opts):
                X.append(feat(o,cur,ctx)); Y.append(1.0 if i==chosen else 0.0); G.append(gid)
            gid+=1
X=np.array(X,dtype=np.float32); Y=np.array(Y,dtype=np.float32); G=np.array(G,dtype=np.int64)
print(f'parsed {nfiles} games -> {gid} decisions, {len(X)} option-rows, {X.shape[1]} features')
print(f'positive rate {Y.mean():.3f} (≈1/avg-options)')
np.savez_compressed('_bc_data.npz', X=X, Y=Y, G=G)
print('saved _bc_data.npz')

"""BC v2 extract: same scalar features PLUS a card-id index per option (for an embedding),
so the net can learn card-SPECIFIC behavior (play Enhanced Hammer, use Dudunsparce, etc.)."""
import sys, json, glob, os; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from cg.api import OptionType, CardType
from agent.cards import get_db
db=get_db(); DL=r'C:\Users\Henrique\Downloads'
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
def scalars(o,state,ctx):
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
# pass 1: build card vocab
files=[f for f in glob.glob(os.path.join(DL,'*.json')) if '-0' not in f and '-1' not in f]
X=[]; C=[]; Y=[]; G=[]; vocab={None:0}; gid=0; ng=0
for fp in files:
    try: d=json.load(open(fp,encoding='utf-8'))
    except Exception: continue
    if not isinstance(d,dict) or 'steps' not in d: continue
    ng+=1
    for step in d['steps']:
        for seat,ag in enumerate(step):
            obs=ag.get('observation') or {}; sel=obs.get('select'); cur=obs.get('current'); act=ag.get('action')
            if not sel or not cur or cur.get('yourIndex')!=seat: continue
            opts=sel.get('option') or []
            if len(opts)<2 or not isinstance(act,list) or len(act)!=1: continue
            ch=act[0]
            if not (0<=ch<len(opts)): continue
            ctx=sel.get('context',0)
            for i,o in enumerate(opts):
                cid=card_id(o,cur)
                if cid not in vocab: vocab[cid]=len(vocab)
                X.append(scalars(o,cur,ctx)); C.append(vocab[cid]); Y.append(1.0 if i==ch else 0.0); G.append(gid)
            gid+=1
X=np.array(X,np.float32); C=np.array(C,np.int64); Y=np.array(Y,np.float32); G=np.array(G,np.int64)
print(f'{ng} games -> {gid} decisions, {len(X)} rows, {X.shape[1]} scalars, vocab={len(vocab)} cards')
np.savez_compressed('_bc_data2.npz', X=X, C=C, Y=Y, G=G, vocab_size=len(vocab))
import pickle; pickle.dump(vocab, open('_bc_vocab.pkl','wb'))
print('saved _bc_data2.npz + _bc_vocab.pkl')

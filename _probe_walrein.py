"""Does the heuristic agent use Walrein's POTENTIAL (the lock/disruption), or just attack?
Tracks: reaches Walrein, Frigid Fangs (the lock), Megaton Fall (170 closer), Dudunsparce
draw-ability, Enhanced/Crushing Hammer (energy strip). Win% is NOT an ELO predictor."""
import sys, random; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from cg.api import OptionType
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose, _card_id_from_option
from agent.base import read_deck
from agent.cards import get_db, validate_deck
db=get_db()
W=read_deck('deck_cand_walrein.csv')
ok,prob=validate_deck(W); print(f'walrein deck: {len(W)} cards valid={ok} {prob}')
WALREIN,DUDUNSPARCE=943,66
FRIGID=[a.attackId for a in db.attacks_of(WALREIN) if 'Frigid' in a.name][0]
MEGATON=[a.attackId for a in db.attacks_of(WALREIN) if 'Megaton' in a.name][0]
HAMMERS={1081,1120}  # Enhanced Hammer, Crushing Hammer
print(f'Frigid Fangs={FRIGID}, Megaton Fall={MEGATON}')
def greedy(deck,seed):
    r=random.Random(seed)
    def f(obs): return list(deck) if obs.get('select') is None else choose(obs,rng=r)
    return f
def has(p,cid):
    a=p.get('active') or []
    if a and a[0] and a[0].get('id')==cid: return True
    return any(b and b.get('id')==cid for b in (p.get('bench') or []))
def run(name,opp,n=50):
    a=greedy(W,1); b=greedy(opp,2); wins=games=0
    rw=frig=mega=dud=ham=0
    for g in range(n):
        a0=(g%2==0); me=0 if a0 else 1
        obs,_=battle_start(W if a0 else opp, opp if a0 else W)
        if obs is None: continue
        agents=(a,b) if a0 else (b,a); steps=0
        rW=fF=mF=dD=hH=False
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
                    for idx in ch:
                        opts=sel.get('option') or []
                        if 0<=idx<len(opts):
                            o=opts[idx]
                            t=o.get('type')
                            if t==OptionType.ATTACK.value:
                                if o.get('attackId')==FRIGID: fF=True
                                elif o.get('attackId')==MEGATON: mF=True
                            elif t==OptionType.ABILITY.value:
                                ipa=o.get('inPlayArea'); ipi=o.get('inPlayIndex'); mp=cur['players'][me]
                                tgt=(mp.get('active') or [None])[0] if ipa==4 else (mp.get('bench') or [])[ipi] if (ipa==5 and ipi is not None and ipi<len(mp.get('bench') or [])) else None
                                if tgt and tgt.get('id')==DUDUNSPARCE: dD=True
                            elif t==OptionType.PLAY.value and _card_id_from_option(o,cur) in HAMMERS:
                                hH=True
                    obs=battle_select(ch)
                else:
                    obs=battle_select(agents[who](obs))
                steps+=1
                if steps>30000: break
        finally: battle_finish()
        games+=1
        rw+=rW; frig+=fF; mega+=mF; dud+=dD; ham+=hH
    print(f'  vs {name:<10}: {wins}/{games}={wins/games:.0%} | reachWalrein {rw}/{games}, FrigidFangs(lock) {frig}/{games}, MegatonFall {mega}/{games}, Dudunsparce-draw {dud}/{games}, Hammer(strip) {ham}/{games}')

print('\nWalrein — does the agent use the lock/disruption potential? (greedy, 50 games; win% != ELO)')
for nm,fn in [('Dragapult','deck_cand_dragapult_real.csv'),('Hops-v2c','deck_cand_hops_hybrid_v2c.csv'),('Alakazam','deck_cand_alakazam_pro.csv')]:
    run(nm, read_deck(fn), 50)

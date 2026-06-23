"""Functional proof for the TR Spidops toolbox deck. Tracks per game whether the agent:
build 4+ TR Pokemon (enables Mewtwo), uses Spidops 'Charging Up' ability, attacks with
Mewtwo / Rocket Rush / Mimikyu. Win rate shown but is NOT an ELO predictor."""
import sys, random; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from cg.api import OptionType
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose
from agent.base import read_deck
from agent.cards import get_db
db=get_db()
TR=read_deck('deck_cand_tr_spidops.csv')
SPIDOPS,MEWTWO,MIMIKYU=401,431,434
# attack ids
ROCKET_RUSH=[a.attackId for a in db.attacks_of(SPIDOPS) if 'Rush' in a.name][0]
ERASURE=[a.attackId for a in db.attacks_of(MEWTWO) if 'Erasure' in a.name][0]
GEMSTONE=612
TR_POKE=set(c for c in set(TR) if db.is_pokemon(c) and 'Rocket' in (db.card(c).name or ''))
print(f'attack ids -> Rocket Rush {ROCKET_RUSH}, Erasure Ball {ERASURE}, Gemstone {GEMSTONE}; TR Pokemon ids {sorted(TR_POKE)}')
def greedy(deck,seed):
    r=random.Random(seed)
    def f(obs): return list(deck) if obs.get('select') is None else choose(obs,rng=r)
    return f
def count_tr_inplay(p):
    n=0
    a=p.get('active') or []
    if a and a[0] and a[0].get('id') in TR_POKE: n+=1
    for b in (p.get('bench') or []):
        if b and b.get('id') in TR_POKE: n+=1
    return n
def run(name,opp,n=50):
    a=greedy(TR,1); b=greedy(opp,2); wins=games=0
    reached4=spid_ab=atk_mewtwo=atk_rush=atk_mimi=0
    for g in range(n):
        a0=(g%2==0); me=0 if a0 else 1
        obs,_=battle_start(TR if a0 else opp, opp if a0 else TR)
        if obs is None: continue
        agents=(a,b) if a0 else (b,a); steps=0
        r4=sa=am=ar=ami=False
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
                    if count_tr_inplay(cur['players'][me])>=4: r4=True
                    ch=agents[who](obs)
                    for idx in ch:
                        opts=sel.get('option') or []
                        if 0<=idx<len(opts):
                            o=opts[idx]
                            if o.get('type')==OptionType.ABILITY.value:
                                ipa=o.get('inPlayArea'); ipi=o.get('inPlayIndex'); mp=cur['players'][me]
                                tgt=(mp.get('active') or [None])[0] if ipa==4 else (mp.get('bench') or [])[ipi] if (ipa==5 and ipi is not None and ipi<len(mp.get('bench') or [])) else None
                                if tgt and tgt.get('id')==SPIDOPS: sa=True
                            if o.get('type')==OptionType.ATTACK.value:
                                aid=o.get('attackId')
                                if aid==ERASURE: am=True
                                elif aid==ROCKET_RUSH: ar=True
                                elif aid==GEMSTONE: ami=True
                    obs=battle_select(ch)
                else:
                    obs=battle_select(agents[who](obs))
                steps+=1
                if steps>30000: break
        finally: battle_finish()
        games+=1
        reached4+=r4; spid_ab+=sa; atk_mewtwo+=am; atk_rush+=ar; atk_mimi+=ami
    print(f'  vs {name:<10}: {wins}/{games}={wins/games:.0%} | 4+TR board {reached4}/{games}, Spidops-ability {spid_ab}/{games}, Mewtwo-atk {atk_mewtwo}/{games}, RocketRush {atk_rush}/{games}, Mimikyu {atk_mimi}/{games}')

print('\nTR Spidops — functional execution (greedy, 50 games each; win% NOT an ELO predictor):')
for nm,fn in [('Dragapult','deck_cand_dragapult_real.csv'),('Hops-v2c','deck_cand_hops_hybrid_v2c.csv'),('Alakazam','deck_cand_alakazam_pro.csv')]:
    run(nm, read_deck(fn), 50)

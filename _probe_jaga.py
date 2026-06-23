"""Functional proof: does the agent execute jaga-Starmie's key decisions?
Tracks, per game, whether the agent: evolves Staryu->Mega Starmie, evolves the
Duskull->Dusclops->Dusknoir line, and USES Munkidori's Adrena-Brain ability.
Reports win rate too, but with the standing caveat: win rate != real ELO."""
import sys, random; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from cg.api import OptionType
from cg.game import battle_start, battle_select, battle_finish
from agent.policy import choose
from agent.base import read_deck
from agent.cards import get_db
db=get_db()
JAGA=read_deck('deck_cand_starmie_jaga.csv')
STARYU,MEGASTARMIE=1030,1031
DUSKULL,DUSCLOPS,DUSKNOIR=131,132,133
MUNKIDORI=112
def greedy(deck,seed):
    r=random.Random(seed)
    def f(obs): return list(deck) if obs.get('select') is None else choose(obs,rng=r)
    return f
def inplay_ids(p):
    out=set()
    a=p.get('active') or []
    if a and a[0]: out.add(a[0].get('id'))
    for b in (p.get('bench') or []):
        if b: out.add(b.get('id'))
    return out
def run(opp_name, opp_deck, n=50):
    a=greedy(JAGA,1); b=greedy(opp_deck,2)
    wins=games=0
    ev_starmie=ev_dusclops=ev_dusknoir=munki_ability=mega_attack=0
    for g in range(n):
        a0=(g%2==0); me=0 if a0 else 1
        obs,_=battle_start(JAGA if a0 else opp_deck, opp_deck if a0 else JAGA)
        if obs is None: continue
        agents=(a,b) if a0 else (b,a); steps=0
        seen_starmie=seen_dusclops=seen_dusknoir=used_munki=used_mega=False
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
                    ids=inplay_ids(cur['players'][me])
                    if MEGASTARMIE in ids: seen_starmie=True
                    if DUSCLOPS in ids: seen_dusclops=True
                    if DUSKNOIR in ids: seen_dusknoir=True
                    ch=agents[who](obs)
                    for idx in ch:
                        opts=sel.get('option') or []
                        if 0<=idx<len(opts):
                            o=opts[idx]
                            if o.get('type')==OptionType.ABILITY.value:
                                # Munkidori ability: inPlayArea/idx -> is it Munkidori?
                                ipa=o.get('inPlayArea'); ipi=o.get('inPlayIndex'); mp=cur['players'][me]
                                tgt=(mp.get('active') or [None])[0] if ipa==4 else (mp.get('bench') or [])[ipi] if (ipa==5 and ipi is not None and ipi<len(mp.get('bench') or [])) else None
                                if tgt and tgt.get('id')==MUNKIDORI: used_munki=True
                            if o.get('type')==OptionType.ATTACK.value and o.get('attackId') in (1487,1488):
                                used_mega=True
                    obs=battle_select(ch)
                else:
                    obs=battle_select(agents[who](obs))
                steps+=1
                if steps>30000: break
        finally: battle_finish()
        games+=1
        ev_starmie+=seen_starmie; ev_dusclops+=seen_dusclops; ev_dusknoir+=seen_dusknoir
        munki_ability+=used_munki; mega_attack+=used_mega
    print(f'  vs {opp_name:<10}: {wins}/{games}={wins/games:.0%}  | reached MegaStarmie {ev_starmie}/{games}, Dusclops {ev_dusclops}/{games}, Dusknoir {ev_dusknoir}/{games}; used Munkidori-ability {munki_ability}/{games}, MegaStarmie-attack {mega_attack}/{games}')

print('JAGA Mega Starmie — functional execution (greedy, 50 games each):')
print('(win rate is NOT a real-ELO predictor — the point is the execution counts)')
for nm,fn in [('Dragapult','deck_cand_dragapult_real.csv'),('Hops-v2c','deck_cand_hops_hybrid_v2c.csv'),('Alakazam','deck_cand_alakazam_pro.csv')]:
    run(nm, read_deck(fn), 50)

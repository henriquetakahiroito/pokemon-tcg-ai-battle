"""Run the v2c deck under whatever policy.py is currently in place, vs the field.
Called twice (old policy, then new policy) to isolate the PILOT effect on v2c."""
import sys, random; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from selfplay.baselines import read_deck
from selfplay.harness import play_match
from agent.policy import choose
class G:
    def __init__(s,deck,seed): s.deck=deck; s.r=random.Random(seed)
    def __call__(s,obs): return list(s.deck) if obs.get('select') is None else choose(obs,rng=s.r)
V2C=read_deck('deck_cand_hops_hybrid_v2c.csv')
opps=[('Dragapult','deck_cand_dragapult_real.csv'),('teaparty','deck_cand_teaparty.csv'),
      ('Lucario','deck_cand_lucario_riolu.csv'),('Alakazam','deck_cand_alakazam_pro.csv'),
      ('Crustle','deck_crustle.csv')]
tag=sys.argv[1] if len(sys.argv)>1 else '?'
print(f'[{tag} policy] v2c greedy, 60 games each:')
tot_w=tot=0
for name,fn in opps:
    res=play_match(G(V2C,1),G(read_deck(fn),2),n_games=60,alternate=True)
    print(f'  v2c vs {name:<10}: {res.winrate_a():.0%}  ({res.wins_a}-{res.wins_b})')
    tot_w+=res.wins_a; tot+=res.wins_a+res.wins_b
print(f'  OVERALL: {tot_w}/{tot} = {tot_w/tot:.0%}')

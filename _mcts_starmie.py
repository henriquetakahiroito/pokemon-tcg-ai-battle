import sys, os, time
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0,'.')
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent
STAR=read_deck('deck_cand_starmie.csv')
for name,opp in [('Dragapult','deck_cand_dragapult_real.csv'),('mirror_hops','deck_cand_hops_hybrid_v2c.csv')]:
    res=play_match(MctsAgent(deck=STAR,seed=1), GreedyAgent(deck=read_deck(opp),seed=2), n_games=14, alternate=True)
    print(f'MCTS Starmie vs {name}: {res.wins_a}W-{res.wins_b}L {res.winrate_a():.0%}')

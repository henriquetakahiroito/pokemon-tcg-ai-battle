"""Offline: does the BC rollout beat the heuristic rollout inside MCTS? Run BOTH and compare.
  python _test_bc_mcts.py            # heuristic rollout
  BC_ROLLOUT=1 python _test_bc_mcts.py   # BC-clone rollout
Leave running; paste only the final line back."""
import sys,os,time; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent
V2C=read_deck('deck_cand_hops_hybrid_v2c.csv'); DRAG=read_deck('deck_cand_dragapult_real.csv')
tag='BC-rollout' if os.environ.get('BC_ROLLOUT')=='1' else 'heuristic-rollout'
t=time.time(); res=play_match(MctsAgent(deck=V2C,seed=1),GreedyAgent(deck=DRAG,seed=2),n_games=20,alternate=True)
print(f'[{tag}] MCTS v2c vs Dragapult: {res.wins_a}-{res.wins_b} = {res.winrate_a():.0%}  ({time.time()-t:.0f}s)')

"""Is my agent piloting like Shumpei (74%, the top Archaludon pilot)? Run my CURRENT agent vs the
real meta, capture its action profile, and print it next to Shumpei's real-game numbers."""
import sys, os, collections
sys.path.insert(0, '.')
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent
from agent.policy import _card_id_from_option
from cg.api import OptionType
EV, AT = OptionType.EVOLVE.value, OptionType.ATTACK.value
_MD, _IRON, _ARCH = 253, 225, 190

evolves = collections.Counter(); attacks = collections.Counter()
reached = [0, 0]            # [games_reached_MD, games]
first_md_steps = []

class Probe(MctsAgent):
    _step = 0; _got = False
    def decide(self, obs):
        pick = super().decide(obs)
        try:
            sel = obs["select"]; opts = sel["option"]; st = obs["current"]
            self._step += 1
            for i in pick:
                if i >= len(opts): continue
                o = opts[i]; t = o.get("type")
                if t == EV:
                    cid = _card_id_from_option(o, st)
                    evolves['archex' if cid == _ARCH else 'other'] += 1
                elif t == AT:
                    aid = o.get("attackId"); attacks[aid] += 1
                    if aid in (_MD, _IRON) and not self._got:
                        self._got = True; first_md_steps.append(self._step)
        except Exception:
            pass
        return pick

MW = read_deck('deck_archaludon_mw.csv')
META = [('megastarmie',26),('hops',15),('megalucario',11),('alakazam',9),('hopsclefairy',9),('dragapult',7)]
print("my agent (mw, current) vs REAL meta — action profile")
for nm, _ in META:
    opp = read_deck(f'deck_meta_real_{nm}.csv')
    for g in range(4):
        hero = Probe(deck=MW, seed=g); hero._got = False; hero._step = 0
        foe = GreedyAgent(deck=opp, seed=g+50)
        r = play_match(hero, foe, n_games=1, alternate=False)
        reached[1] += 1
        if hero._got: reached[0] += 1

tot = sum(attacks.values()) or 1
md = attacks[_MD]+attacks[_IRON]
ev_ok = evolves['archex']; ev_bad = evolves['other']
fm = sorted(first_md_steps)
med = fm[len(fm)//2] if fm else None
print(f"\n{'metric':<26}{'MY AGENT':>12}{'SHUMPEI':>12}")
print(f"{'Metal Defender share':<26}{md/tot:>11.0%}{'53%':>12}")
print(f"{'evolve->Archaludon ex':<26}{ev_ok:>5} ok/{ev_bad} bad{'  121/0':>10}")
print(f"{'reached Metal Defender':<26}{reached[0]}/{reached[1]} ={reached[0]/max(1,reached[1]):>4.0%}{'88%':>9}")
print(f"{'top attacks':<26}{str(attacks.most_common(4))}")

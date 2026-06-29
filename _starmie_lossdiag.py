"""Decode HOW current agent loses to real MegaStarmie — find the residual flaw to encode.
Plays N games; for LOSSES records: reached Metal Defender, fired Raging Hammer finisher(224),
energy-starved turns (active Archaludon <3 energy), capped the tank, prize diff at end."""
import sys, collections
sys.path.insert(0, '.')
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent
from agent.policy import my_state, opp_state, active_of, total_energy
from cg.api import OptionType
AT = OptionType.ATTACK.value
_MD, _IRON, _ARCH, _DURA, _CAPE = 253, 225, 190, 169, 1159

class P(MctsAgent):
    def reset(self):
        self.md = False; self.rage = False; self.starve = 0; self.turns = 0; self.caped = False
    def decide(self, obs):
        pick = super().decide(obs)
        try:
            st = obs["current"]; opts = obs["select"]["option"]
            mp = my_state(st); a = active_of(mp)
            if a and a.get("id") in (_ARCH, _DURA, 170, 840):
                self.turns += 1
                if total_energy(a) < 3: self.starve += 1
                if (a.get("maxHp", 0) or 0) >= 400: self.caped = True
            for i in pick:
                if i < len(opts) and opts[i].get("type") == AT:
                    aid = opts[i].get("attackId")
                    if aid in (_MD, _IRON): self.md = True
                    if aid == 224: self.rage = True
        except Exception:
            pass
        return pick

MW = read_deck('deck_archaludon_mw.csv'); STAR = read_deck('deck_meta_real_megastarmie.csv')
N = 16
wins = losses = 0
loss_diag = collections.Counter(); loss_n = 0
for g in range(N):
    hero = P(deck=MW, seed=g); hero.reset()
    foe = GreedyAgent(deck=STAR, seed=g + 500)
    r = play_match(hero, foe, n_games=1, alternate=(g % 2 == 0))
    won = r.wins_a >= 1
    if won: wins += 1
    else:
        losses += 1; loss_n += 1
        if hero.md: loss_diag['reached_MD'] += 1
        if hero.rage: loss_diag['fired_RagingHammer'] += 1
        if hero.caped: loss_diag['caped_tank'] += 1
        if hero.turns and hero.starve / hero.turns > 0.5: loss_diag['energy_starved'] += 1
print(f"vs real MegaStarmie: {wins}-{losses} = {wins/N:.0%}  ({N} games)")
print(f"\nLOSS diagnostics ({loss_n} losses):")
for k in ('reached_MD', 'fired_RagingHammer', 'caped_tank', 'energy_starved'):
    print(f"   {k:<20} {loss_diag[k]}/{loss_n}")

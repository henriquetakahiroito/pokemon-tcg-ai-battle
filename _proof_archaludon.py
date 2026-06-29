"""LEARNING PROOF: tap the MCTS pilot's REAL in-game choices.

Subclass MctsAgent, record every option it actually picks across full games vs the
meta field, then tally the engine-critical actions:

  EVOLVE  Duraludon(169) -> Archaludon ex(190)   = did it bring the engine online?
  ATTACK  Metal Defender(253) / Iron Blaster(225) = is it swinging the real attacker?
          vs Duraludon hammer(223/224)            = the "hitting with Duraludon" failure

If the agent learned the deck: high evolve rate + attacks dominated by 253/225, the
Duraludon hammers near zero. That is the measurable rebuttal to "HE IS HITTING WITH
DURALUDON" -- not vibes, a distribution.
"""
import sys, os, collections
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
ROOT = r'F:\Claude\pokemon-tcg-agent'
sys.path.insert(0, ROOT)
from selfplay.baselines import read_deck, GreedyAgent
from selfplay.harness import play_match
from agent.agent import MctsAgent
from agent.policy import _card_id_from_option
from cg.api import OptionType

MW = read_deck(os.path.join(ROOT, "deck_archaludon_mw.csv"))
OPPS = {
    "lucario":   "deck_cand_lucario_riolu.csv",
    "dragapult": "deck_cand_dragapult_real.csv",
    "starmie":   "deck_cand_starmie_jaga.csv",
}
_DURALUDON, _ARCH_EX = 169, 190
_MD, _IRON = 253, 225
_HAMMER = (223, 224)
EV_T, AT_T = OptionType.EVOLVE.value, OptionType.ATTACK.value

evolves = collections.Counter()   # target card id -> count
attacks = collections.Counter()   # attackId -> count
hammer_split = collections.Counter()  # 'forced' (no Arch ex in hand) vs 'lazy' (had it, chipped)
from agent.policy import _hand_ids as _HAND_IDS
# opportunity: how many of our turns had Archaludon ex available to evolve into but we held?
evolve_offered = collections.Counter()  # 'offered' / 'taken'

class ProbedMcts(MctsAgent):
    def decide(self, obs):
        pick = super().decide(obs)
        try:
            sel = obs["select"]; opts = sel["option"]; state = obs["current"]
            # tally evolve-into-Archaludon-ex availability this decision
            for o in opts:
                if o.get("type") == EV_T and _card_id_from_option(o, state) == _ARCH_EX:
                    evolve_offered['offered'] += 1
                    break
            for i in pick:
                if i >= len(opts):
                    continue
                o = opts[i]; t = o.get("type")
                if t == EV_T:
                    cid = _card_id_from_option(o, state)
                    if cid is not None:
                        evolves[cid] += 1
                        if cid == _ARCH_EX:
                            evolve_offered['taken'] += 1
                elif t == AT_T:
                    aid = o.get("attackId")
                    if aid is not None:
                        attacks[aid] += 1
                        if aid in (223, 224):
                            act = (state["players"][0].get("active") or [{}])
                            act_id = (act[0] or {}).get("id") if act else None
                            if act_id == _DURALUDON:   # raw Duraludon swing = the real stall
                                had = _ARCH_EX in _HAND_IDS(state)
                                hammer_split['lazy' if had else 'forced'] += 1
                            else:
                                hammer_split['evolved_memdive'] += 1  # legit Archaludon Raging Hammer
        except Exception:
            pass
        return pick

print(f"LEARNING PROOF: MCTS pilot real-game actions (deck_archaludon_mw)\n{'='*60}")
for name, fn in OPPS.items():
    opp = read_deck(os.path.join(ROOT, fn))
    hero = ProbedMcts(deck=MW, seed=1)
    foe = GreedyAgent(deck=opp, seed=101)
    res = play_match(hero, foe, n_games=8, alternate=True)
    print(f"  played {name:<10}: {res.wins_a}-{res.wins_b}")

def nm(cid):
    return {169:"Duraludon",190:"Archaludon ex",840:"Archaludon(baby)",695:"Mega Mawile",
            57:"Relicanth",1071:"Meowth ex",140:"Fez ex"}.get(cid, str(cid))
def an(aid):
    return {253:"Metal Defender",225:"Iron Blaster",223:"Duraludon-hammer",
            224:"Raging Hammer",1006:"Gobble Down",1007:"Huge Bite",840:"Coated Attack"}.get(aid,str(aid))

print(f"\n{'='*60}\nEVOLVES taken (card the agent evolved INTO):")
tot_ev = sum(evolves.values()) or 1
for cid, c in evolves.most_common():
    print(f"   {nm(cid):<18} {c:>4}  ({c/tot_ev:.0%})")
off = evolve_offered['offered']; tk = evolve_offered['taken']
print(f"\n  Archaludon ex evolve uptake: {tk}/{off} decisions where it was offered "
      f"= {tk/off:.0%}" if off else "\n  (Archaludon ex evolve never offered?!)")

print(f"\nATTACKS thrown (attackId distribution):")
tot_at = sum(attacks.values()) or 1
for aid, c in attacks.most_common():
    print(f"   {an(aid):<18} {c:>4}  ({c/tot_at:.0%})")
md_share = (attacks[_MD]+attacks[_IRON]) / tot_at
ham_share = (attacks[223]+attacks[224]) / tot_at
print(f"\n  REAL attacker share (Metal Defender + Iron Blaster): {md_share:.0%}")
print(f"  Duraludon hammer share (the failure mode):           {ham_share:.0%}")
print(f"  VERDICT: {'LEARNED — swings the engine, not Duraludon' if md_share > ham_share else 'STILL stalling on Duraludon'}")
f = hammer_split['forced']; l = hammer_split['lazy']
ev = hammer_split['evolved_memdive']
print(f"\n  Hammer breakdown: raw-Duraludon forced(no Arch ex in hand)={f}  raw-Duraludon lazy(had Arch ex)={l}  evolved-Memory-Dive(legit)={ev}")
print(f"  => {l} avoidable stalls" if l else "  => 0 avoidable stalls: every raw-Duraludon swing was a forced chip while digging")

"""Ceruledge ex — deterministic discard-aggro rule agent.

Same school as agent/lucario.py (pure rules, no MCTS / no value net): score every legal option
per SelectContext, pick the best. Ceruledge is a linear aggro deck whose ENTIRE skill collapses
to one idea the generic engine can't learn but a rule states in one line:

    Abyssal Flames = 30 + 20 * (Energy cards in YOUR discard pile), for 1 Fire energy.

So energy in the discard pile IS damage. Unlike Archaludon (which CAPS Metal-in-discard at 2 for
Raging Hammer), Ceruledge wants the discard pile as FULL of energy as possible — uncapped. The
discard / search sub-decisions (what to pitch to Ultra Ball / Carmine, what to fetch) are the only
real choices; everything else is "evolve Charcadet -> Ceruledge ex, attach 1 Fire, swing".

Plan:
  - Lead/develop Charcadet, evolve to Ceruledge ex ASAP.
  - Keep ONE Fire energy on Ceruledge (Abyssal costs 1 Fire); extra Fire is wasted on it.
  - Pitch energy (Fighting first — 14 of them vs 6 Fire) to the discard to pump Abyssal Flames.
  - Solrock (Cosmic Beam 70, needs Lunatone benched) / Lunatone (Power Gem 50) are the secondary
    engine attackers while Ceruledge sets up.
  - Raging Amethyst (447) needs Fire+Psychic+Metal — uncastable here, so it never enters the plan.
"""
from __future__ import annotations

from collections import defaultdict

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType, Pokemon,
    SelectContext, all_card_data, to_observation_class,
)

from .base import BaseAgent


class C:
    CHARCADET = 319
    CERULEDGE_EX = 320
    LUNATONE = 675
    SOLROCK = 676
    DRILBUR = 81
    FIRE_ENERGY = 2
    FIGHTING_ENERGY = 6
    EXPLORER = 1185
    CARMINE = 1192
    BOSS_ORDERS = 1182
    ULTRA_BALL = 1121
    FIGHTING_GONG = 1142
    POKEMON_CATCHER = 1124
    NIGHT_STRETCHER = 1097
    POKE_PAD = 1152
    BRILLIANT_BLENDER = 1128
    LEGACY_ENERGY = 12


ABYSSAL_FLAMES = 446
RAGING_AMETHYST = 447
COSMIC_BEAM = 980
POWER_GEM = 979
SAND_SPRAY = 96
LOW_DECK_COUNT = 8

_DRAW_SUPPORTERS = {C.EXPLORER, C.CARMINE}
_ENERGY_CARD_TYPES = {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}

_all_card = all_card_data()
card_table = {card.cardId: card for card in _all_card}


def _is_energy(cid):
    data = card_table.get(cid)
    return bool(data) and data.cardType in _ENERGY_CARD_TYPES


class AttackPlan:
    def __init__(self, attacker=-1, target=-1, attack_index=-1, remain_hp=-1, needs_energy=False):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


def get_card(obs, area, index, player_index):
    player = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return player.hand[index]
        case AreaType.DISCARD:
            return player.discard[index]
        case AreaType.ACTIVE:
            return player.active[index]
        case AreaType.BENCH:
            return player.bench[index]
        case AreaType.PRIZE:
            return player.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon):
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    return max(0, count)


def target_score(pokemon):
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    if pokemon.id in {173, 174, 190, 1071}:  # low-value support/pivot
        score -= 200
    if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


class CeruledgePolicy:
    def __init__(self, obs, plan, ability_used):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)
        self.plan = plan
        self.ability_used = ability_used

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.energy_in_discard = 0
        self.fire_in_hand = 0
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.ceruledge_ready = False     # a Ceruledge ex in play already has >=1 energy
        self.have_ceruledge_line = False  # Charcadet or Ceruledge ex somewhere in play
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self):
        if not self.select.option or self.select.maxCount == 0:
            return [], self.plan, self.ability_used
        if self.context == SelectContext.MAIN:
            self._plan_attack()
        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked[: self.select.maxCount], self.plan, self.ability_used

    # ---- board scan -------------------------------------------------------
    def _count_cards(self):
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in {C.CHARCADET, C.CERULEDGE_EX}:
                self.have_ceruledge_line = True
            if pokemon.id == C.CERULEDGE_EX and len(pokemon.energies) >= 1:
                self.ceruledge_ready = True
        for card in self.me.hand:
            self.hand_counts[card.id] += 1
            if card.id == C.FIRE_ENERGY:
                self.fire_in_hand += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1
            if _is_energy(card.id):
                self.energy_in_discard += 1

    def _scan_main_options(self):
        if self.context != SelectContext.MAIN:
            return
        for option in self.select.option:
            if option.type == OptionType.PLAY:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if card.id == C.BOSS_ORDERS:
                    self.can_gust = True
            elif option.type == OptionType.RETREAT:
                self.can_switch = True
            elif option.type == OptionType.ATTACK:
                self.can_attack = True

    def _my_board(self):
        return self.me.active + self.me.bench

    def _opponent_board(self):
        return self.opponent.active + self.opponent.bench

    def _low_deck(self):
        return self.me.deckCount <= LOW_DECK_COUNT

    def _effect_id(self):
        eff = getattr(self.select, "effect", None)
        return eff.id if eff else None

    # ---- attack plan ------------------------------------------------------
    def _abyssal_damage(self):
        return 30 + 20 * self.energy_in_discard

    def _base_attack(self, pokemon, attack_index):
        """Return (energy_required, base_damage, base_score) or None."""
        if pokemon.id == C.CERULEDGE_EX:
            if attack_index == 0:
                return 1, self._abyssal_damage(), 0
            return None  # Raging Amethyst (447) is uncastable in a Fire/Fighting deck
        if attack_index == 1:
            return None
        if pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            return 1, 70, 0
        if pokemon.id == C.LUNATONE:
            return 2, 50, 0
        if pokemon.id == C.DRILBUR:
            return 2, 20, 0
        return None

    def _plan_attack(self):
        best_score = -1
        self.plan = AttackPlan()
        if self.state.turn < 2:
            return
        for attacker_index, my_pokemon in enumerate(self._my_board()):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break
            for attack_index in range(2):
                attack = self._base_attack(my_pokemon, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack
                energy_count = len(my_pokemon.energies)
                needs_energy = False
                if energy_count < energy_required:
                    if self.hand_counts[C.FIRE_ENERGY] + self.hand_counts[C.FIGHTING_ENERGY] >= 1 \
                            and not self.state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue
                for target_index, op_pokemon in enumerate(self._opponent_board()):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break
                    op_data = card_table[op_pokemon.id]
                    damage = base_damage
                    if op_data.weakness == EnergyType.FIRE and my_pokemon.id == C.CERULEDGE_EX:
                        damage *= 2
                    elif op_data.weakness == EnergyType.FIGHTING and my_pokemon.id in {C.SOLROCK, C.LUNATONE, C.DRILBUR}:
                        damage *= 2
                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        score *= damage / op_pokemon.hp
                    if len(self.opponent.prize) <= prize:
                        score = 50000
                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    # prefer the real attacker (Ceruledge) over chip engine attacks at equal value
                    score += 80 if my_pokemon.id == C.CERULEDGE_EX else 0
                    if score > best_score:
                        best_score = score
                        self.plan = AttackPlan(
                            attacker=attacker_index, target=target_index,
                            attack_index=attack_index, remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    # ---- option scoring ---------------------------------------------------
    def _score_option(self, option):
        if option.type == OptionType.NUMBER:
            return option.number
        if option.type == OptionType.YES:
            return 100 if self.context == SelectContext.IS_FIRST else 1
        if option.type == OptionType.NO:
            return 0
        if option.type == OptionType.CARD:
            return self._score_card_choice(option)
        if option.type == OptionType.PLAY:
            return self._score_play(option)
        if option.type == OptionType.ATTACH:
            return self._score_attach(option)
        if option.type == OptionType.EVOLVE:
            return self._score_evolve(option)
        if option.type == OptionType.ABILITY:
            return 30000
        if option.type == OptionType.RETREAT:
            return 2000 if self.plan.attacker >= 1 else -1
        if option.type == OptionType.ATTACK:
            # Only attack when _plan_attack found a real attack (plan.attacker >= 0). Engine chip
            # attacks (Sand Spray etc.) that aren't part of the plan score below END so we develop
            # instead of wasting a turn. Still picked if it's genuinely the only legal action.
            if self.plan.attacker < 0:
                return -1
            return 1000
        return 0

    def _score_card_choice(self, option):
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return 0
        if self.context in {SelectContext.SWITCH, SelectContext.TO_ACTIVE}:
            return self._score_active_choice(option, card)
        if self.context == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if self.context == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if self.context in {SelectContext.DISCARD, SelectContext.DISCARD_ENERGY,
                            SelectContext.DISCARD_ENERGY_CARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD}:
            return self._score_discard(card)
        if self.context == SelectContext.TO_HAND_ENERGY:
            return self._score_to_hand_energy(card)
        if self.context == SelectContext.ATTACH_FROM and isinstance(card, Pokemon):
            return self._energy_target_score(card, option.area == AreaType.ACTIVE)
        return 0

    def _score_active_choice(self, option, card):
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex != self.my_index:
            return 100 if option.index == self.plan.target - 1 else 0
        score = len(card.energies) * 2
        if option.index == self.plan.attacker - 1:
            score += 100
        if card.id == C.CERULEDGE_EX:
            score += 30
        elif card.id == C.CHARCADET:
            score += 8
        elif card.id == C.SOLROCK:
            score += 5
        elif card.id == C.LUNATONE:
            score += 4
        return score

    def _score_setup_active(self, card):
        # Lead a Charcadet so we can evolve into Ceruledge ex; Solrock if no Charcadet.
        if card.id == C.CHARCADET:
            return 5
        if card.id == C.SOLROCK:
            return 3
        if card.id == C.LUNATONE:
            return 2
        if card.id == C.DRILBUR:
            return 1
        return 0

    def _score_to_hand(self, card):
        """Ultra Ball / search fetch: complete the Ceruledge line first, then fuel / engine."""
        cid = card.id
        if cid == C.CERULEDGE_EX:
            return 30000 if self.field_counts[C.CHARCADET] >= 1 and self.field_counts[C.CERULEDGE_EX] == 0 else 24000
        if cid == C.CHARCADET:
            line = self.field_counts[C.CHARCADET] + self.field_counts[C.CERULEDGE_EX]
            return 28000 if line == 0 else (8000 if line == 1 else -100)
        if cid == C.FIRE_ENERGY:
            # need a Fire to actually swing Abyssal; grab one if Ceruledge can't yet attack
            return 22000 if (not self.ceruledge_ready and self.fire_in_hand == 0) else 3000
        if cid == C.LUNATONE and self.field_counts[C.LUNATONE] == 0:
            return 15000
        if cid == C.SOLROCK and self.field_counts[C.SOLROCK] == 0:
            return 14000
        if cid == C.DRILBUR:
            return 6000
        if cid in _DRAW_SUPPORTERS:
            return 12000 - self.hand_counts[cid] * 4000
        if cid == C.ULTRA_BALL and self.hand_counts[cid] == 0:
            return 9000
        if cid == C.FIGHTING_ENERGY:
            return 2000
        return 5000

    def _score_to_hand_energy(self, card):
        # recovering energy to hand: take Fire if we still need to power Ceruledge
        if card.id == C.FIRE_ENERGY and not self.ceruledge_ready and self.fire_in_hand == 0:
            return 200
        return 50

    def _score_discard(self, card):
        """THE skill. Abyssal Flames = 30 + 20*energy-in-discard (uncapped), so energy in the
        discard is damage — pitch it freely. Keep ONE Fire to actually attack, lean on the 14
        Fighting energy as fuel, and never throw away the last evo piece or last draw supporter."""
        cid = card.id
        line = self.field_counts[C.CHARCADET] + self.field_counts[C.CERULEDGE_EX]
        # Fighting energy: plentiful (14), pure fuel -> best thing to pitch.
        if cid == C.FIGHTING_ENERGY:
            return 9000
        # Fire energy: also fuels Abyssal, but we must keep >=1 to pay the attack cost.
        if cid == C.FIRE_ENERGY:
            keep_fire = (not self.ceruledge_ready) and self.fire_in_hand <= 1
            return -2000 if keep_fire else 7000
        # special energy (Legacy etc.) still counts as discard fuel
        if _is_energy(cid):
            return 6000
        # never pitch the only line piece if we have no Ceruledge in play yet
        if cid == C.CERULEDGE_EX:
            return -5000 if self.field_counts[C.CERULEDGE_EX] == 0 else 4000
        if cid == C.CHARCADET:
            return -4000 if line == 0 else (1000 if line == 1 else 5000)
        # keep the last draw supporter; surplus is fine to pitch
        if cid in _DRAW_SUPPORTERS:
            return 5500 if self.hand_counts[cid] > 1 else -1500
        # engine / utility: pitch duplicates, keep singletons-ish
        if cid in {C.LUNATONE, C.SOLROCK}:
            return 3000 if self.field_counts[cid] >= 1 else -500
        # dead / duplicate trainers are fine fodder
        return 4000 if self.hand_counts[cid] > 1 else 2000

    def _score_play(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        data = card_table[card.id]
        if data.cardType == CardType.POKEMON:
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card):
        if card.id in {C.LUNATONE, C.SOLROCK} and self.field_counts[card.id] >= 1:
            return -1
        if card.id == C.CHARCADET and self.field_counts[C.CHARCADET] + self.field_counts[C.CERULEDGE_EX] >= 3:
            return -1
        return 20000

    def _score_play_trainer(self, card):
        cid = card.id
        if cid == C.BOSS_ORDERS:
            return 3200 if self.plan.target >= 1 else -1
        if cid == C.POKEMON_CATCHER:
            return 3100 if self.plan.target >= 1 else -1
        if cid == C.ULTRA_BALL:
            # dig for the line, or just to load energy into discard
            need_line = self.field_counts[C.CERULEDGE_EX] == 0
            return 3400 if need_line else 2600
        if cid in _DRAW_SUPPORTERS:
            return -1 if self._low_deck() else 3000
        if cid == C.FIGHTING_GONG:
            return 2800
        if cid == C.NIGHT_STRETCHER:
            return 2400
        if cid == C.POKE_PAD:
            return 2200
        if cid == C.BRILLIANT_BLENDER:
            return 2300
        return 10000

    def _energy_target_score(self, pokemon, active):
        score = 8000 + (10 if active else 0)
        if pokemon.id == C.CERULEDGE_EX:
            # exactly ONE Fire powers Abyssal; more on Ceruledge is wasted (extra energy is better
            # in the discard fueling damage), so only prioritise it until it has its first.
            score += 300 if len(pokemon.energies) < 1 else -400
        elif pokemon.id == C.SOLROCK:
            score += 60 if len(pokemon.energies) < 1 else -100
        elif pokemon.id == C.LUNATONE:
            score += 40 if len(pokemon.energies) < 2 else -100
        elif pokemon.id == C.CHARCADET:
            score -= 200  # don't waste energy on the unevolved basic
        return score

    def _score_attach(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        score = self._energy_target_score(pokemon, option.inPlayArea == AreaType.ACTIVE)
        # prefer attaching Fire to Ceruledge, Fighting to the engine attackers
        if card.id == C.FIRE_ENERGY and pokemon.id == C.CERULEDGE_EX:
            score += 150
        if card.id == C.FIGHTING_ENERGY and pokemon.id in {C.SOLROCK, C.LUNATONE}:
            score += 80
        board_index = option.inPlayIndex if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if board_index == self.plan.attacker and self.plan.needs_energy:
            score += 200
        return score

    def _score_evolve(self, option):
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0
        # Charcadet -> Ceruledge ex: top priority, this is the whole deck.
        return 12000 + len(pokemon.energies)


class CeruledgeAgent(BaseAgent):
    name = "ceruledge"

    def __init__(self, deck=None, seed=None, **_):
        super().__init__(deck, seed)
        self._pre_turn = -1
        self._ability_used = False
        self._plan = AttackPlan()

    def decide(self, obs_dict):
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            self._pre_turn = -1
            self._ability_used = False
            self._plan = AttackPlan()
            return list(self.deck)
        if self._pre_turn != obs.current.turn:
            self._pre_turn = obs.current.turn
            self._ability_used = False
            self._plan = AttackPlan()
        sel, self._plan, self._ability_used = CeruledgePolicy(
            obs, self._plan, self._ability_used).choose()
        return sel

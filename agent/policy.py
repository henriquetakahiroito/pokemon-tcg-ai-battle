"""Heuristic action policy.

`option_scores` assigns a desirability score to every option of a decision;
`choose` turns those scores into a contract-valid selection (right count, unique,
in range). Used directly by the greedy baseline and as the rollout/prior policy
inside MCTS. It is deliberately fast and side-effect free (no engine calls).
"""
from __future__ import annotations

import random

from cg.api import OptionType, SelectContext
from .cards import get_db
from .observation import my_state, opp_state, active_of, total_energy
from .evaluate import attack_damage_estimate

# Contexts where selecting MORE is good (take the max count).
_BENEFICIAL_COUNT = {
    SelectContext.DRAW_COUNT.value,
    SelectContext.REMOVE_DAMAGE_COUNTER_COUNT.value,
    SelectContext.DAMAGE_COUNTER_COUNT.value,
    SelectContext.TO_HAND.value,
    SelectContext.HEAL.value,
    SelectContext.DAMAGE.value,
    SelectContext.DAMAGE_COUNTER.value,
    SelectContext.DAMAGE_COUNTER_ANY.value,
    SelectContext.REMOVE_DAMAGE_COUNTER.value,
    SelectContext.TO_BENCH.value,
}
# Contexts where selecting is a cost (take the min count).
_COSTLY_COUNT = {
    SelectContext.DISCARD.value,
    SelectContext.DISCARD_ENERGY.value,
    SelectContext.DISCARD_ENERGY_CARD.value,
    SelectContext.DISCARD_TOOL_CARD.value,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD.value,
    SelectContext.TO_DECK.value,
    SelectContext.TO_DECK_BOTTOM.value,
    SelectContext.TO_DECK_ENERGY.value,
    SelectContext.TO_PRIZE.value,
}


def _card_id_from_option(o: dict, state: dict) -> int | None:
    """Resolve the card id an option refers to, when locatable from the state."""
    area = o.get("area")
    idx = o.get("index")
    pi = o.get("playerIndex")
    if idx is None:
        return None
    # PLAY options reference a card in the acting player's HAND by index alone (no `area`
    # field). Without this, EVERY hand-play resolved to None and fell back to a flat 5.0,
    # silently disabling all bespoke play scoring (Boss's Orders, Crushing Hammer, Lillie's
    # sequencing, Clefairy/Meowth gating, Mega Signal, Salvatore, ...).
    if area is None:
        if o.get("type") == OptionType.PLAY.value:
            area = 2  # HAND
        else:
            return None
    try:
        player = state["players"][pi if pi is not None else state["yourIndex"]]
    except (KeyError, IndexError, TypeError):
        return None
    # AreaType: 1=DECK 2=HAND 3=DISCARD 4=ACTIVE 5=BENCH 6=PRIZE 12=LOOKING ...
    if area == 4:  # ACTIVE
        active = player.get("active") or []
        return active[0].get("id") if active and active[0] is not None else None
    area_key = {1: "deck", 2: "hand", 3: "discard", 5: "bench", 12: "looking"}.get(area)
    if area_key:
        arr = player.get(area_key) or []
        if 0 <= idx < len(arr) and arr[idx] is not None:
            return arr[idx].get("id")
    return None


def _get_pokemon_from_option(o: dict, state: dict) -> dict | None:
    """Return the in-play Pokémon dict an option refers to (ACTIVE or BENCH)."""
    area = o.get("area")
    idx = o.get("index")
    pi = o.get("playerIndex")
    if area is None:
        return None
    try:
        player = state["players"][pi if pi is not None else state["yourIndex"]]
    except (KeyError, IndexError, TypeError):
        return None
    if area == 4:  # ACTIVE
        active = player.get("active") or []
        return active[0] if active and active[0] is not None else None
    if area == 5:  # BENCH
        bench = player.get("bench") or []
        if idx is not None and 0 <= idx < len(bench) and bench[idx] is not None:
            return bench[idx]
    return None


# ---- Hops deck card IDs ----
_HOPS_CRAMORANT_ID    = 311   # Fickle Spitting: 120 dmg, only at opp 3-4 prizes
_HOPS_PHANTUMP_ID     = 878   # Splashing Dodge: 10 dmg + protection flip (early game)
_HOPS_TREVENANT_ID    = 879   # Horrifying Revenge / Corner (post-KO revenge attacker)
_HOPS_SNORLAX_ID      = 304   # bench ability "Extra Helpings": Hop's attacks +30 (no stack)
_DUNSPARCE_ID         = 65    # Basic; evolves into Dudunsparce
_DUDUNSPARCE_ID       = 66    # bench ability "Run Away Draw": draw 3 then bounce to deck
_POKE_PAD_ID          = 1152  # Item: search a non-Rule-Box Pokémon to hand (Dudunsparce/Snorlax)
_HOPS_CHOICE_BAND_ID  = 1171  # tool: -1 energy cost AND +30 dmg, Hop's only
_BROCKS_SCOUTING_ID   = 1210  # supporter: search 2 basics OR 1 evolution to hand
_MEOWTH_EX_ID         = 1071  # Basic ex: on bench-play, Last-Ditch Catch = search a Supporter
                              # to hand (consistency tutor). 2-prize bait → keep on bench.
# Supporters that shuffle/discard your HAND then draw — must be played AFTER your items,
# never before, or you shuffle your own setup away.
_LILLIE_DET_ID        = 1227  # Lillie's Determination: shuffle hand → draw 6 (8 at 6 prizes)
_JUDGE_ID             = 1213  # Judge: both players shuffle hand → draw 4
_HARLEQUIN_ID         = 1223  # Harlequin: shuffle hand → flip, draw 5/3
_HAND_SHUFFLE_DRAWS   = {_LILLIE_DET_ID, _JUDGE_ID, _HARLEQUIN_ID}

# Attack IDs for Hops Pokémon
_FICKLE_SPITTING     = 433   # Cramorant 120 dmg, conditional
_SPLASHING_DODGE     = 1266  # Phantump  10 dmg + protection
_HORRIFYING_REVENGE  = 1267  # Trevenant 30+ dmg, scales with Hop's KOs taken
_CORNER              = 1268  # Trevenant 90 dmg + locks retreat

_MIST_ENERGY_ID = 11  # Protects active from effects of opponent's attacks
_LEGACY_ENERGY_ID = 12  # provides any type; the Pokémon it's on takes +1 Prize on KO — a
                        # prize-race finisher: put on an attacker when opp has 3-4 prizes left
_TELEPATH_PSYCHIC_ENERGY_ID = 19  # on a Psychic Pokémon, search a Psychic Basic to the bench
                                  # (free board development) + powers Psychic attackers

# Dragapult meta counter cards
_LILLIE_CLEFAIRY_EX_ID = 272   # bench ability "Fairy Zone": Dragon Pokémon get Psychic weakness
_POSTWICK_ID           = 1255  # stadium: Hop's attacks +30 (also overrides Watchtower)
_WATCHTOWER_ID         = 1256  # Team Rocket's Watchtower: Colorless Pokémon have no abilities

_DRAGON_ENERGY_TYPE  = 9
_PSYCHIC_ENERGY_TYPE = 5

# ---- Slowking / Kyurem "Seek Inspiration" combo deck ----
# Slowking's Seek Inspiration ({P}{C}) discards the top deck card and, if it is a
# non-Rule-Box Pokémon, copies one of ITS attacks for free. We set the top card with
# Academy at Night (hand->top) or Ciphermaniac (search->top), then fire Seek to copy a
# board-wiping attack. Engine support for this copy is confirmed (see _probe_seek2.py).
_SLOWKING_ID       = 163
_KYUREM_ID         = 144   # copy target: Trifrost = 110 dmg to 3 of opp's Pokémon
_MIMIKYU_ID        = 434   # copy target: Gemstone Mimicry = copy opp's ACTIVE Tera attack
_ANNIHILAPE_ID     = 224   # copy target: Destined Fight = both Active Pokémon KO'd
_ACADEMY_NIGHT_ID  = 1248  # stadium: put a card from hand on top of deck (combo enabler)
_CIPHERMANIAC_ID   = 1188  # supporter: search 2, put on top in any order (combo enabler)
_SEEK_INSPIRATION  = 213   # Slowking attack id — the copy attack
_TRIFROST          = 188   # Kyurem attack id (110 to 3, effect-text damage)
_GEMSTONE_MIMICRY  = 612   # Mimikyu attack id — copies opp's active Tera Pokémon's attack
_PHANTOM_DIVE      = 154   # Dragapult ex attack (200) — what Mimicry copies vs Dragapult
_ANNI_DESTINED     = 305   # both active KO'd (effect)
_COMBO_PAYOFFS = (_KYUREM_ID, _MIMIKYU_ID, _ANNIHILAPE_ID)

# ---- Mega Starmie ex / Cinderace deck (the real-ladder #1) ----
# Cinderace opens via its Explosiveness ability (face-down active at setup), then Turbo
# Flare accelerates 3 Basic Energy to the bench each turn to power up Mega Starmie ex,
# which hits 120+50 for ONE Water energy (Jetting Blow) or 210 unmodifiable (Nebula Beam).
_CINDERACE_ID    = 666
_STARYU_ID       = 1030
_MEGA_STARMIE_ID = 1031
_TURBO_FLARE     = 965    # Cinderace: 50 dmg + search 3 Basic Energy to bench (accel engine)
_JETTING_BLOW    = 1487   # Mega Starmie: 120 + 50 to bench, {W}
_NEBULA_BEAM     = 1488   # Mega Starmie: 210, unaffected by Weakness/Resistance/effects
_IGNITION_ENERGY_ID = 17  # provides {C}{C}{C} ONLY on an Evolution Pokémon; on a Basic it
                          # gives just {C} and is discarded at end of turn (wasted)
# Mega Starmie engine trainers (scored only while piloting the deck — see _starmie_in_play).
_BUDDY_POFFIN_ID     = 1086  # Item: search 2 Basics (≤70 HP) to bench — Staryu fetcher
_MEGA_SIGNAL_ID      = 1145  # Item: tutor a Mega Evolution ex (Mega Starmie)
_SALVATORE_ID        = 1189  # Supporter: search an ability-less evolution (Mega Starmie/Cinderace line)
_HILDA_ID            = 1225  # Supporter: search an Evolution Pokémon + an Energy
_POKEGEAR_ID         = 1122  # Item: dig top 7 for a Supporter
_WALLYS_COMPASSION_ID = 1229  # Supporter: full-heal 1 Mega Evolution ex
_CRUSHING_HAMMER_ID  = 1120  # Item: coin-flip discard an opponent's energy
_BOSS_ORDERS_ID      = 1182  # Supporter: gust an opponent's benched Pokémon active
_HEROS_CAPE_ID       = 1159  # Tool: +100 HP
_NIGHT_STRETCHER_ID  = 1097  # Item: recover a Pokémon or Basic Energy from discard
_ULTRA_BALL_ID       = 1121  # Item: discard 2 → search any Pokémon

# ---- Mega Lucario ex deck (Fighting; structurally Hops-like: cheap engine attack + big
# finisher + damage modifiers + a built-in gust). Weak to Psychic. ----
_RIOLU_ID            = 677
_MEGA_LUCARIO_ID     = 678   # Stage 1, 340 HP, weak {P}
_AURA_JAB            = 982   # {F} 130 + attach up to 3 Basic {F} from discard to bench (engine)
_MEGA_BRAVE          = 983   # {F}{F} 270, can't be used the following turn (finisher)
_FIRE_ENERGY_ID      = 6     # Basic {F} Energy
_HARIYAMA_ID         = 674   # ability on evolve: gust opp's benched active (built-in Boss's)
_LUNATONE_ID         = 675   # ability Lunar Cycle: discard {F} → draw 3 (needs Solrock)
_SOLROCK_ID          = 676
_PREMIUM_POWER_PRO_ID = 1141  # Item: {F} attacks do +30 to the Active this turn
_FIGHTING_GONG_ID    = 1142  # Item: search a Basic {F} Energy or Basic {F} Pokémon
_CARMINE_ID          = 1192  # Supporter: (going first, T1 usable) discard hand, draw 5
_DUSK_BALL_ID        = 1102  # Item: look at bottom 7, take a Pokémon
_GRAVITY_MOUNTAIN_ID = 1252  # Stadium: every Stage 2 in play gets -30 HP

# Tool IDs that reduce attack energy cost by 1 (Hop's Pokémon only).
_COST_REDUCE_TOOLS = {_HOPS_CHOICE_BAND_ID}


def _tool_cost_reduction(pokemon: dict) -> int:
    """Energy saved on attacks by attached tools. Choice Band only works on Hop's Pokémon."""
    cid = pokemon.get("id")
    if cid is None:
        return 0
    db = get_db()
    if not db.name(cid).startswith("Hop's"):
        return 0
    tools = pokemon.get("tools") or []
    return sum(1 for t in tools if (t or {}).get("id") in _COST_REDUCE_TOOLS)


def _hops_dmg_bonus(pokemon: dict, state: dict) -> int:
    """Extra damage Hop's Pokémon get from all active modifiers.

    Sources (all apply before Weakness/Resistance):
    - Snorlax bench ability "Extra Helpings": +30 (doesn't stack with a second Snorlax)
    - Postwick stadium: +30
    - Hop's Choice Band (tool on attacker): +30
    Maximum possible bonus: +90
    """
    if pokemon is None:
        return 0
    db = get_db()
    if not db.name(pokemon.get("id", 0)).startswith("Hop's"):
        return 0
    bonus = 0
    me = state["yourIndex"]
    mp = state["players"][me]
    # Snorlax "Extra Helpings" bench ability (+30, no stacking)
    bench = mp.get("bench") or []
    if any((b or {}).get("id") == _HOPS_SNORLAX_ID for b in bench):
        bonus += 30
    # Postwick stadium (+30)
    stadium = state.get("stadium") or []
    if any((c or {}).get("id") == _POSTWICK_ID for c in stadium):
        bonus += 30
    # Hop's Choice Band attached to attacker (+30)
    for t in (pokemon.get("tools") or []):
        if (t or {}).get("id") == _HOPS_CHOICE_BAND_ID:
            bonus += 30
            break
    return bonus


def _opp_prizes(state: dict) -> int:
    op = state["players"][1 - state["yourIndex"]]
    return len(op.get("prize") or [])


def _score_attack(o: dict, state: dict) -> float:
    db = get_db()
    me = state["yourIndex"]
    mp = state["players"][me]
    op = state["players"][1 - me]
    my_act = active_of(mp)
    op_act = active_of(op)
    aid = o.get("attackId")
    ai = db.attack(aid) if aid is not None else None
    base = ai.damage if ai else 0
    op_pr = _opp_prizes(state)
    our_kos = 6 - op_pr  # how many of our Pokémon have been KO'd

    # The Dunsparce draw line must NEVER attack — it only evolves and draws (Run Away Draw).
    # If it's stuck Active we retreat it (see _score_retreat / _score_attach), never attack.
    if my_act and my_act.get("id") in (_DUNSPARCE_ID, _DUDUNSPARCE_ID):
        return -500.0

    # Meowth ex must NEVER attack — it is a consistency tutor only (Last-Ditch Catch fetches
    # a Supporter when bricked / to grab Boss's Orders for a gust). Retreat it if stuck Active.
    if my_act and my_act.get("id") == _MEOWTH_EX_ID:
        return -500.0

    # ---- Slowking combo: copied-attack valuation ----
    # These attacks (Seek Inspiration's copy targets) carry their damage in effect text,
    # so the DB damage field is 0 and the generic scorer would undervalue them badly.
    if aid == _GEMSTONE_MIMICRY:
        # Mimikyu: copy the opponent's ACTIVE Tera Pokémon's attack (e.g. Dragapult's
        # Phantom Dive). Only works vs an active Tera; we proxy that with Dragon-type active.
        # The copied Phantom Dive itself is then scored by the general branch (Fairy Zone ×2).
        if _opp_active_is_dragon(state):
            return 90.0 if _fairy_zone_active(state) else 60.0
        return 3.0  # no active Tera to mimic — dead
    if aid == _PHANTOM_DIVE:
        # Phantom Dive used via the Mimikyu copy: fall through to the general scorer below so
        # the Fairy Zone ×2 (Slowking is Psychic vs Dragon) and KO detection apply.
        pass
    # ---- Cinderace = ACCEL engine, NOT the attacker. The #1 pilot (keidroid) uses Turbo
    # Flare only 2-3× early to bank energy onto a benched Mega Starmie, then attacks with
    # Starmie. Our agent was mashing Turbo Flare 30× (vs 16 Starmie attacks) because the old
    # flat 60 outranked evolving Staryu->Mega Starmie (~12) and charging it — so Starmie never
    # assembled. Turbo Flare must score BELOW advancing the Starmie line. ----
    if aid == _TURBO_FLARE:
        # The #1 pilots (keidroid, Yushin Ito 69-10) Turbo Flare only ~0.3×/game — Cinderace
        # is pure accel, not the attacker. Turbo Flare ENDS the turn, so it must score BELOW
        # every setup play (Mega Signal 18, Salvatore 30/12, Hilda 17, Poffin 16, evolve,
        # attach, draw) so the agent develops Staryu->Mega Starmie + charges it FIRST, then
        # Turbo Flares only as the idle terminal action. A flat 40 here was ending turns before
        # setup and capping our setup rate ~33% vs Yushin's ~76% on the SAME deck.
        mp_ = my_state(state)
        megas = [p for p in [active_of(mp_)] + list(mp_.get("bench") or [])
                 if p and p.get("id") == _MEGA_STARMIE_ID]
        # A Mega Starmie with >=3 energy can Nebula Beam: STOP acceling, bring it active.
        if any(total_energy(p) >= 3 for p in megas):
            return 4.0
        # Low fallback: above END(3) so we still accel when idle, but below all setup/draw
        # plays so it never preempts assembling the Starmie line.
        return 7.0
    # ---- Mega Starmie attack discipline: this is what separates the #1 pilot (Nebula
    # Beam 644×) from the 594-ELO pilot (Jetting Blow spam). Nebula Beam's 210 is the
    # reliable default KO button — UNAFFECTED by weakness/resistance/effects (damage
    # prevention/reduction, Tera walls). Use Jetting Blow only when its cheaper 120
    # already KOs the active (saves energy + the +50 snipes a benched threat). ----
    if aid == _NEBULA_BEAM:
        if op_act is not None:
            hp = op_act.get("hp", 9999)
            if 210 >= hp:
                return 102.0 + 210 / 100.0   # clean, unconditional KO
            return 78.0 + 210 / 100.0        # heavy reliable chip (no whiff vs walls/effects)
        return 70.0
    if aid == _JETTING_BLOW:
        if my_act and op_act:
            dmg = attack_damage_estimate(my_act["id"], op_act["id"], 120)
            if dmg >= op_act.get("hp", 9999):
                return 103.5 + dmg / 100.0   # efficient KO + bench snipe: best line here
            return 55.0 + dmg / 10.0         # only chips — Nebula Beam's 210 is preferred
        return 40.0

    # ---- Mega Lucario attack discipline (Hops-like): Aura Jab is the cheap repeatable
    # engine (130 + accelerate 3 Fire from discard to the bench, no cooldown); Mega Brave
    # is the 270 finisher you save for what Aura Jab can't KO (and it locks itself next turn).
    if aid == _AURA_JAB:
        if my_act and op_act:
            dmg = attack_damage_estimate(my_act["id"], op_act["id"], 130)
            if dmg >= op_act.get("hp", 9999):
                return 104.0 + dmg / 100.0   # cheap KO + bench accel + no cooldown = best line
            accel = 18.0 if _lucario_accel_useful(state) else 0.0
            return 50.0 + dmg / 10.0 + accel
        return 45.0
    if aid == _MEGA_BRAVE:
        if my_act and op_act:
            dmg = attack_damage_estimate(my_act["id"], op_act["id"], 270)
            if dmg >= op_act.get("hp", 9999):
                return 102.0 + dmg / 100.0   # the finisher: use when Aura Jab's 130 can't KO
            return 60.0 + dmg / 10.0
        return 55.0

    if aid == _ANNI_DESTINED:
        # Both active KO'd: great when we trade Slowking (1 prize) into a 2-prize ex / fat wall.
        oa = active_of(op)
        if oa is not None:
            try:
                c = db.card(oa.get("id"))
                worth = 2.0 if (c and getattr(c, "ex", False)) else 1.0
            except Exception:
                worth = 1.0
            return 38.0 + 22.0 * worth
        return 34.0
    if aid == _TRIFROST:
        return 72.0  # 110 to 3 of opp's Pokémon — board wipe / heavy spread
    if aid == _SEEK_INSPIRATION:
        # Strong when Academy at Night is up (we can guarantee a payoff on top this turn).
        # Without that setup it can whiff, so keep it just above a blind attack.
        return 52.0 if _academy_in_play(state) else 24.0

    # ---- Hops deck phase-aware attack scoring ----

    # Fickle Spitting (Cramorant): 120 dmg for 1 energy — but ONLY at opp 3-4 prizes.
    if ai and ai.attackId == _FICKLE_SPITTING:
        if op_pr not in (3, 4):
            return -300.0  # does literally NOTHING outside the 3-4 prize window — never fire it
        # Vs Lucario: only fire at a target the 120 actually KOs for value (Solrock / Lunatone /
        # damaged Hariyama). Never waste it on the 340 HP Mega Lucario.
        if _opp_is_lucario(state) and not _cramorant_target_ok(state):
            return -300.0
        # In the window: exceptional value — 120 dmg for 1 energy, buy setup time
        return 80.0

    # Splashing Dodge (Phantump): 10 dmg + coin-flip full protection.
    # Valuable early (stall + protect setup); weak later when we need real damage.
    if ai and ai.attackId == _SPLASHING_DODGE:
        # Vs Lucario, Phantump IS the gameplan: stall + protect with Splashing Dodge while we
        # set up. But once we've taken a KO, step aside (low) so the freshly promoted Phantump
        # EVOLVES to Trevenant and Horrifying Revenge (130 + modifiers, 2x Psychic) KOs the Mega.
        if _opp_is_lucario(state):
            return 40.0 if our_kos == 0 else 6.0
        if op_pr == 6:
            return 35.0  # early game: protection > damage
        return 18.0  # late game: very low damage, deprioritize

    # Horrifying Revenge (Trevenant): 30 base + 100 if a Hop's Pokémon was KO'd by attack
    # damage during opponent's last turn (binary, not per-KO). Approximation: any KO taken.
    # Alakazam's "Powerful Hand" places damage counters (effect, not attack damage) so it
    # does NOT trigger this condition.
    if ai and ai.attackId == _HORRIFYING_REVENGE:
        actual_base = 130 if our_kos > 0 else 30
        if my_act and op_act:
            bonus = _hops_dmg_bonus(my_act, state)
            dmg = attack_damage_estimate(my_act["id"], op_act["id"], actual_base + bonus)
            # Fairy Zone: Trevenant (Psychic) 2× vs Dragon
            if _fairy_zone_active(state) and _opp_active_is_dragon(state):
                dmg *= 2
            if dmg >= op_act.get("hp", 9999):
                return 100.0 + dmg / 100.0
            return 30.0 + dmg / 10.0
        return 45.0 if our_kos > 0 else 20.0

    # Corner (Trevenant): 90 dmg + locks retreat.
    if ai and ai.attackId == _CORNER:
        if my_act and op_act:
            bonus = _hops_dmg_bonus(my_act, state)
            dmg = attack_damage_estimate(my_act["id"], op_act["id"], 90 + bonus)
            # Fairy Zone: Trevenant (Psychic type) gets 2× vs Dragon opponent
            if _fairy_zone_active(state) and _opp_active_is_dragon(state):
                dmg *= 2
            if dmg >= op_act.get("hp", 9999):
                return 100.0 + dmg / 100.0
            return 38.0 + dmg / 10.0  # +8 vs base for retreat-lock utility
        return 25.0

    # Horrifying Revenge fairy zone check is above; also apply for the general case.
    # ---- General attack scoring with Hops damage modifiers ----
    if my_act and op_act and base > 0:
        bonus = _hops_dmg_bonus(my_act, state)
        dmg = attack_damage_estimate(my_act["id"], op_act["id"], base + bonus)
        # Fairy Zone: if attacker is Psychic type and opponent is Dragon, apply 2× weakness
        if _fairy_zone_active(state) and _opp_active_is_dragon(state):
            try:
                if db.card(my_act["id"]).energyType == _PSYCHIC_ENERGY_TYPE:
                    dmg *= 2
            except Exception:
                pass
        if dmg >= op_act.get("hp", 9999):
            return 100.0 + dmg / 100.0
        return 30.0 + dmg / 10.0
    return 20.0


_XEROSIC_ID              = 1197  # Xerosic's Machinations: discard opponent to 3 cards
_SECRET_BOX_ID           = 1092  # Item: discard 3 → search Item+Tool+Supporter+Stadium
_TR_TRANSCEIVER_ID       = 1134  # Item: search for a Team Rocket's Supporter


def _fairy_zone_active(state: dict) -> bool:
    """True if Lillie's Clefairy ex is on our bench (Fairy Zone ability: Dragon → Psychic weakness)."""
    mp = my_state(state)
    bench = mp.get("bench") or []
    return any((b or {}).get("id") == _LILLIE_CLEFAIRY_EX_ID for b in bench)


def _opp_active_is_dragon(state: dict) -> bool:
    """True if opponent's active Pokémon is Dragon type (Dragapult ex, etc.)."""
    db = get_db()
    op = opp_state(state)
    op_act = active_of(op)
    if op_act is None:
        return False
    cid = op_act.get("id")
    if cid is None:
        return False
    try:
        return db.card(cid).energyType == _DRAGON_ENERGY_TYPE
    except Exception:
        return False


def _opp_has_dragon_threat(state: dict) -> bool:
    """True if opponent has any Dragon-type Pokémon in play (active or bench).

    Checks bench too so we can prepare Clefairy when Dreepy/Drakloak are visible
    but Dragapult hasn't come active yet.
    """
    db = get_db()
    op = opp_state(state)
    op_act = active_of(op)
    if op_act is not None:
        cid = op_act.get("id")
        if cid is not None:
            try:
                if db.card(cid).energyType == _DRAGON_ENERGY_TYPE:
                    return True
            except Exception:
                pass
    for b in (op.get("bench") or []):
        if b is None:
            continue
        cid = b.get("id")
        if cid is None:
            continue
        try:
            if db.card(cid).energyType == _DRAGON_ENERGY_TYPE:
                return True
        except Exception:
            pass
    return False


def _slowking_in_play(state: dict) -> bool:
    """True if we have a Slowking in play (active or bench) — i.e. piloting the combo."""
    mp = my_state(state)
    if (active_of(mp) or {}).get("id") == _SLOWKING_ID:
        return True
    return any((b or {}).get("id") == _SLOWKING_ID for b in (mp.get("bench") or []))


def _slowking_active_charged(state: dict) -> bool:
    """True if our active is a Slowking with enough energy to fire Seek Inspiration ({P}{C})."""
    act = active_of(my_state(state))
    if not act or act.get("id") != _SLOWKING_ID:
        return False
    return total_energy(act) >= 2


def _academy_in_play(state: dict) -> bool:
    return any((c or {}).get("id") == _ACADEMY_NIGHT_ID for c in (state.get("stadium") or []))


def _opp_active_is_basic(state: dict) -> bool:
    """True if opponent's active Pokémon is a Basic (Axe Blast auto-KOs Basics)."""
    db = get_db()
    oa = active_of(opp_state(state))
    if oa is None:
        return False
    try:
        c = db.card(oa.get("id"))
        return bool(c and getattr(c, "basic", False))
    except Exception:
        return False


def _best_combo_payoff(state: dict) -> int:
    """Pick which copy target to put on top of deck, given the opponent's active.

    - Active Tera Dragon (Dragapult ex) -> Mimikyu: copies its own Phantom Dive back at it;
      with Lillie's Clefairy on our bench (Fairy Zone) Slowking is Psychic so it's ×2 = OHKO.
      Even without Clefairy, that's 200 to the active + 6 counters on their bench.
    - Big evolution wall -> Annihilape (Destined Fight trades Slowking 1-prize into it).
    - otherwise -> Kyurem (Trifrost spreads 110 to 3; note Tera Pokémon on the Bench are immune).
    """
    db = get_db()
    if _opp_active_is_dragon(state):
        return _MIMIKYU_ID
    oa = active_of(opp_state(state))
    if oa is not None:
        try:
            c = db.card(oa.get("id"))
            if c is not None and not getattr(c, "basic", False) and (oa.get("hp", 0) or 0) > 160:
                return _ANNIHILAPE_ID
        except Exception:
            pass
    return _KYUREM_ID


def _starmie_in_play(state: dict) -> bool:
    """True if we're piloting the Mega Starmie deck (Cinderace/Staryu/Mega Starmie in play)."""
    mp = my_state(state)
    ids = {(active_of(mp) or {}).get("id")}
    ids.update((b or {}).get("id") for b in (mp.get("bench") or []))
    return bool(ids & {_CINDERACE_ID, _STARYU_ID, _MEGA_STARMIE_ID})


def _mega_starmie_ready(state: dict) -> bool:
    """True if a Mega Starmie ex is in play with at least 1 energy (can Jetting Blow)."""
    mp = my_state(state)
    for p in [active_of(mp)] + list(mp.get("bench") or []):
        if p and p.get("id") == _MEGA_STARMIE_ID and total_energy(p) >= 1:
            return True
    return False


def _mega_starmie_in_play(state: dict) -> bool:
    mp = my_state(state)
    for p in [active_of(mp)] + list(mp.get("bench") or []):
        if p and p.get("id") == _MEGA_STARMIE_ID:
            return True
    return False


def _mega_starmie_damaged(state: dict) -> bool:
    mp = my_state(state)
    for p in [active_of(mp)] + list(mp.get("bench") or []):
        if p and p.get("id") == _MEGA_STARMIE_ID:
            if (p.get("hp", 0) or 0) < (p.get("maxHp", 0) or 0):
                return True
    return False


def _opp_has_energy(state: dict) -> bool:
    """True if any opponent Pokémon has energy attached (Crushing Hammer has a target)."""
    op = opp_state(state)
    for p in [active_of(op)] + list(op.get("bench") or []):
        if p and total_energy(p) >= 1:
            return True
    return False


def _opp_bench_ko_available(state: dict) -> bool:
    """True if an opponent benched Pokémon is worth gusting up for a KO: an ex (2 prizes)
    or anything Nebula Beam (210) can one-shot. Used to value Boss's Orders."""
    op = opp_state(state)
    db = get_db()
    for b in (op.get("bench") or []):
        if not b:
            continue
        hp = b.get("hp", 9999)
        try:
            is_ex = bool(db.card(b.get("id")).ex)
        except Exception:
            is_ex = False
        if hp <= 210 and (is_ex or hp <= 130):
            return True
    return False


def _boss_positive_prize_ko(state: dict) -> bool:
    """True if gusting an opponent benched Pokémon gives a POSITIVE prize trade — i.e. a 2-prize
    ex we can realistically KO (our Hop's attackers reach ~150 with Snorlax/Band/Postwick). Used
    to gate fetching Boss's Orders via Meowth ex: only do it when the gust actually wins prizes,
    not for an even 1-for-1."""
    op = opp_state(state)
    db = get_db()
    for b in (op.get("bench") or []):
        if not b:
            continue
        try:
            is_ex = bool(db.card(b.get("id")).ex)
        except Exception:
            is_ex = False
        if is_ex and b.get("hp", 9999) <= 200:
            return True
    return False


def _hand_ids(state: dict) -> list:
    mp = my_state(state)
    return [(h or {}).get("id") for h in (mp.get("hand") or []) if h]


def _have_energy_in_hand(state: dict) -> bool:
    db = get_db()
    return any(db.is_energy(i) for i in _hand_ids(state) if i is not None)


def _hand_has_supporter(state: dict) -> bool:
    """True if any Supporter is already in hand (so we can advance without Meowth's tutor)."""
    db = get_db()
    for i in _hand_ids(state):
        if i is None:
            continue
        ct = db.card_type(i)
        if ct is not None and ct.name == "SUPPORTER":
            return True
    return False


def _hard_forbidden(o: dict, state: dict) -> bool:
    """Moves the MCTS root bandit must NEVER pick. These are rule-breaking but materially
    NEUTRAL (they don't lose the game), so the rollout value won't avoid them and a -500 prior
    is ignored — they must be FILTERED OUT of the candidate list (see MctsAgent.decide). Covers:
      - Hop's Choice Band on anything other than an attacking Hop's Pokémon (Phantump/Cramorant/
        Trevenant). Replay 81913147 showed it on Meowth ex / Dunsparce / Dudunsparce.
      - Any energy attached to the Dunsparce draw line (it must never carry energy)."""
    if o.get("type") != OptionType.ATTACH.value:
        return False
    db = get_db()
    cid = _card_id_from_option(o, state)  # the energy/tool being attached (from hand)
    mp = my_state(state)
    ipa, ipi = o.get("inPlayArea"), o.get("inPlayIndex")
    tgt = None
    if ipa == 4:
        tgt = active_of(mp)
    elif ipa == 5:
        bench = mp.get("bench") or []
        if ipi is not None and 0 <= ipi < len(bench):
            tgt = bench[ipi]
    tid = (tgt or {}).get("id")
    if cid == _HOPS_CHOICE_BAND_ID and tid not in (_HOPS_PHANTUMP_ID, _HOPS_CRAMORANT_ID, _HOPS_TREVENANT_ID):
        return True
    if cid is not None and db.is_energy(cid) and tid in (_DUNSPARCE_ID, _DUDUNSPARCE_ID):
        return True
    return False


def _meowth_bench_score(state: dict, bench_n: int) -> float:
    """When to put Meowth ex on the bench. Its Last-Ditch Catch tutors a Supporter on bench-play,
    but it's a 170 HP 2-prize liability — play it only for a reason. Per the rules:
      - If we already have a Supporter in hand and no special need: DON'T play it (-2).
      - Bricked (no Supporter): try POKEGEAR 3.0 FIRST (cheap Item dig, no 2-prize liability).
        Only commit Meowth if we have no Pokegear to try (or it already whiffed → no longer in hand).
      - Gust: fetch Boss's Orders ONLY when it sets up a POSITIVE prize-trade KO (a 2-prize ex we
        can knock out) — never for an even 1-for-1, and never over-fetch a Boss we already hold.
    Never attacks (handled in _score_attack)."""
    if bench_n >= 5:
        return -1.0
    hand = _hand_ids(state)
    if not _hand_has_supporter(state):
        # bricked: Pokegear first if we have it, else bench Meowth to guarantee a Supporter
        if _POKEGEAR_ID in hand:
            return 3.0    # below Pokegear's score so it plays first; revisit Meowth if it whiffs
        return 15.0
    # we have a Supporter already — the only remaining reason is a gust that WINS prizes: a
    # 2-prize ex we can KO, and only if we aren't already holding a Boss.
    if _boss_positive_prize_ko(state) and (_BOSS_ORDERS_ID not in hand):
        return 14.0       # fetch Boss's Orders for a positive-prize-trade KO
    return -2.0           # don't need it: leave the 2-prize liability in the deck


def _active_is_staryu(state: dict) -> bool:
    return (active_of(my_state(state)) or {}).get("id") == _STARYU_ID


def _staryu_in_play(state: dict) -> bool:
    mp = my_state(state)
    return any((p or {}).get("id") == _STARYU_ID
               for p in [active_of(mp)] + list(mp.get("bench") or []))


def _pokemon_in_play_count(state: dict) -> int:
    mp = my_state(state)
    n = 1 if active_of(mp) else 0
    return n + sum(1 for b in (mp.get("bench") or []) if b)


def _opp_main_attacker(state: dict) -> dict | None:
    """Opponent Pokémon carrying the most energy — their main attacker (Crushing Hammer target)."""
    op = opp_state(state)
    best, best_e = None, -1
    for p in [active_of(op)] + list(op.get("bench") or []):
        if p and total_energy(p) > best_e:
            best, best_e = p, total_energy(p)
    return best if best_e >= 1 else None


def _boss_two_prize_with_jetting(state: dict) -> bool:
    """True if gusting an opponent's benched ex (2 prizes) into the Active Spot lets Jetting
    Blow's 120 (with Weakness) KO it this turn — Boss's Orders converts to 2 prizes."""
    db = get_db()
    mp = my_state(state)
    op = opp_state(state)
    attacker = next((p for p in [active_of(mp)] + list(mp.get("bench") or [])
                     if p and p.get("id") == _MEGA_STARMIE_ID), None)
    atk_id = attacker.get("id") if attacker else _MEGA_STARMIE_ID
    for b in (op.get("bench") or []):
        if not b:
            continue
        try:
            is_ex = bool(db.card(b.get("id")).ex)
        except Exception:
            is_ex = False
        if not is_ex:
            continue
        dmg = attack_damage_estimate(atk_id, b.get("id"), 120)
        if dmg >= b.get("hp", 9999):
            return True
    return False


def _boss_ko_target_available(state: dict) -> int:
    """Boss's Orders gusts an opponent's benched Pokémon active. Return the prize value
    (2 for an ex, 1 otherwise) of the most valuable benched target our ACTIVE attacker
    could KO this turn if it were dragged up; 0 if there is no KO target."""
    db = get_db()
    mp = my_state(state)
    op = opp_state(state)
    act = active_of(mp)
    if not act:
        return 0
    a = db.best_attack(act.get("id"))
    if not a or a.damage <= 0:
        return 0
    bonus = _hops_dmg_bonus(act, state)
    try:
        attacker_psychic = db.card(act.get("id")).energyType == _PSYCHIC_ENERGY_TYPE
    except Exception:
        attacker_psychic = False
    best = 0
    for b in (op.get("bench") or []):
        if not b:
            continue
        dmg = attack_damage_estimate(act.get("id"), b.get("id"), a.damage + bonus)
        # Fairy Zone: once gusted active, a Dragon target is 2× to our Psychic attacker.
        if attacker_psychic and _fairy_zone_active(state):
            try:
                if db.card(b.get("id")).energyType == _DRAGON_ENERGY_TYPE:
                    dmg *= 2
            except Exception:
                pass
        if dmg >= b.get("hp", 9999):
            try:
                is_ex = bool(db.card(b.get("id")).ex)
            except Exception:
                is_ex = False
            best = max(best, 2 if is_ex else 1)
    return best


def _hops_in_play(state: dict) -> bool:
    """True if we're piloting the Hops shell (any Hop's Pokémon / Dunsparce line in play)."""
    mp = my_state(state)
    ids = {(active_of(mp) or {}).get("id")}
    ids.update((b or {}).get("id") for b in (mp.get("bench") or []))
    return bool(ids & {_DUNSPARCE_ID, _DUDUNSPARCE_ID, _HOPS_PHANTUMP_ID,
                       _HOPS_TREVENANT_ID, _HOPS_CRAMORANT_ID, _HOPS_SNORLAX_ID})


def _has_in_play(state: dict, cid: int) -> bool:
    mp = my_state(state)
    return any((p or {}).get("id") == cid for p in [active_of(mp)] + list(mp.get("bench") or []))


def _snorlax_completes_ko(state: dict) -> bool:
    """True if we have no Snorlax in play and its +30 'Extra Helpings' buff would turn our
    Hop's active's attack from non-lethal into a KO on the opponent's active."""
    db = get_db()
    if _has_in_play(state, _HOPS_SNORLAX_ID):
        return False
    mp = my_state(state)
    op = opp_state(state)
    act = active_of(mp)
    oa = active_of(op)
    if not act or not oa or not db.name(act.get("id", 0)).startswith("Hop's"):
        return False
    a = db.best_attack(act.get("id"))
    if not a or a.damage <= 0:
        return False
    bonus = _hops_dmg_bonus(act, state)
    hp = oa.get("hp", 9999)
    cur = attack_damage_estimate(act["id"], oa["id"], a.damage + bonus)
    withlax = attack_damage_estimate(act["id"], oa["id"], a.damage + bonus + 30)
    return cur < hp <= withlax


def _score_poke_pad(state: dict) -> float:
    """Poké Pad fetches a non-Rule-Box Pokémon. Prioritize it when it grabs the piece we
    need now: Dudunsparce (to evolve a Dunsparce → draw engine) or a Snorlax (whose +30
    completes a KO this turn). Otherwise it's a normal consistency item."""
    if not _hops_in_play(state):
        return 7.0
    if _has_in_play(state, _DUNSPARCE_ID) and _DUDUNSPARCE_ID not in _hand_ids(state):
        return 13.0  # fetch Dudunsparce to get the draw engine online
    if _snorlax_completes_ko(state):
        return 13.0  # fetch a Snorlax — its +30 turns this turn's attack into a KO
    return 7.0


def _score_boss_orders(state: dict) -> float:
    """Play Boss's Orders only when it sets up a KO — prefer dragging a 2-prize ex."""
    v = _boss_ko_target_available(state)
    return 12.0 + 4.0 * v if v > 0 else 4.0


def _score_hand_shuffle_draw(state: dict) -> float:
    """Lillie's Determination / Judge / Harlequin shuffle your HAND into the deck, then draw.
    Play them ONLY after you've used your items (and benched your basics) — otherwise they
    shuffle your own setup away. Low while actionable cards remain; high to refill a spent hand."""
    db = get_db()
    actionable = 0
    for i in _hand_ids(state):
        if i is None:
            continue
        ct = db.card_type(i)
        nm = ct.name if ct else ""
        if nm == "ITEM":
            actionable += 1                       # play items first
        elif nm == "POKEMON":
            try:
                if db.card(i).basic:
                    actionable += 1               # bench your basics first
            except Exception:
                pass
    if actionable > 0:
        return 2.5    # below items (7): items / bench development go first
    return 13.0       # hand is spent — refill now (and it's good when the hand is clogged)


def _starmie_play_score(cid: int, state: dict, hand_n: int, bench_n: int):
    """Bespoke play priorities for the Mega Starmie engine, per explicit rules. Returns None
    to fall through to the generic scorer. Only consulted when _starmie_in_play."""
    has_mega = _mega_starmie_in_play(state)
    turn = state.get("turn", 99)

    # Salvatore: the aggressive 2nd-player opening — going SECOND on turn 1 (you may attack)
    # with a Staryu active, search the evolution to evolve Staryu -> Mega Starmie, energize,
    # then attack (Jetting Blow / Nebula Beam per the attack-discipline scorer). Going FIRST
    # you cannot attack T1, so it's only the normal evolution search there.
    if cid == _SALVATORE_ID:
        fp = state.get("firstPlayer", -1)
        going_second = (fp is not None and fp >= 0 and fp != state.get("yourIndex"))
        if turn <= 2 and going_second and _active_is_staryu(state) and not has_mega:
            return 30.0
        return 12.0 if _staryu_in_play(state) and not has_mega else 7.0

    # Mega Signal: tutor the main attacker (Mega Starmie) whenever we don't have one.
    if cid == _MEGA_SIGNAL_ID:
        return 18.0 if not has_mega else 4.0

    # Buddy-Buddy Poffin: develop the board when we have 2 or fewer Pokémon in play.
    if cid == _BUDDY_POFFIN_ID:
        return 16.0 if _pokemon_in_play_count(state) <= 2 else 5.0

    # Hilda: search Evolution + Energy — exactly when we can evolve a Staryu but have no
    # energy in hand to power the resulting Mega Starmie.
    if cid == _HILDA_ID:
        if _staryu_in_play(state) and not has_mega and not _have_energy_in_hand(state):
            return 17.0
        return 8.0

    # Boss's Orders: gust when it sets up a KO (2-prize ex via Jetting Blow is best).
    if cid == _BOSS_ORDERS_ID:
        if _boss_two_prize_with_jetting(state):
            return 20.0
        return _score_boss_orders(state)

    # Crushing Hammer: fire EVERY time the opponent has energy to strip (target chosen below).
    if cid == _CRUSHING_HAMMER_ID:
        return 13.0 if _opp_has_energy(state) else 1.0

    # Hero's Cape: +100 HP — always wants to sit on a Mega Starmie or a Staryu (future Starmie).
    if cid == _HEROS_CAPE_ID:
        return 12.0 if (has_mega or _staryu_in_play(state)) else 3.0

    if cid == _POKEGEAR_ID:
        return 8.0 + max(0, 5 - hand_n)
    if cid == _WALLYS_COMPASSION_ID:
        return 13.0 if _mega_starmie_damaged(state) else 1.0
    if cid == _NIGHT_STRETCHER_ID:
        return 8.0
    if cid == _ULTRA_BALL_ID:
        return 11.0 if not has_mega else 6.0
    return None


def _lucario_in_play(state: dict) -> bool:
    """True if we're piloting the Mega Lucario deck (Riolu / Mega Lucario in play)."""
    mp = my_state(state)
    return any((p or {}).get("id") in (_RIOLU_ID, _MEGA_LUCARIO_ID)
               for p in [active_of(mp)] + list(mp.get("bench") or []))


def _opp_is_lucario(state: dict) -> bool:
    """True if the OPPONENT is the Mega Lucario deck (Riolu / Mega Lucario in play). Used for the
    Hops vs Lucario gameplan: Phantump stalls, Trevenant finishes, Cramorant only snipes support."""
    op = opp_state(state)
    return any((p or {}).get("id") in (_RIOLU_ID, _MEGA_LUCARIO_ID)
               for p in [active_of(op)] + list(op.get("bench") or []))


def _cramorant_target_ok(state: dict) -> bool:
    """Vs Lucario, Cramorant's Fickle Spitting (120) should only fire at a target it actually
    KOs for value — a Solrock or Lunatone (their draw engine), or a Hariyama already damaged
    enough to fall to 120. NEVER the 340 HP Mega Lucario (the 120 is wasted)."""
    oa = active_of(opp_state(state))
    if not oa:
        return False
    cid = oa.get("id")
    if cid in (_SOLROCK_ID, _LUNATONE_ID):
        return True
    if cid == _HARIYAMA_ID:
        hp = oa.get("hp", 9999)
        max_hp = oa.get("maxHp", hp) or hp
        return hp < max_hp and hp <= 120  # damaged AND within Cramorant's 120 KO range
    return False


def _lucario_accel_useful(state: dict) -> bool:
    """Aura Jab attaches up to 3 Basic {F} from DISCARD to the bench — useful only when we
    have Fire energy in the discard AND a benched Pokémon that still needs energy."""
    db = get_db()
    mp = my_state(state)
    fire_in_discard = any((c or {}).get("id") == _FIRE_ENERGY_ID for c in (mp.get("discard") or []))
    if not fire_in_discard:
        return False
    for b in (mp.get("bench") or []):
        if b and _attack_energy_need(b, db) > 0:
            return True
    return False


def _premium_secures_ko(state: dict) -> bool:
    """True if Premium Power Pro's +30 turns a non-lethal Mega Lucario attack into a KO on
    the opponent's active (130->160 Aura Jab, or 270->300 Mega Brave)."""
    mp = my_state(state)
    op = opp_state(state)
    act = active_of(mp)
    oa = active_of(op)
    if not act or act.get("id") != _MEGA_LUCARIO_ID or not oa:
        return False
    hp = oa.get("hp", 9999)
    aid = act.get("id")
    for base in (130, 270):
        no_boost = attack_damage_estimate(aid, oa.get("id"), base)
        boosted = attack_damage_estimate(aid, oa.get("id"), base + 30)
        if no_boost < hp <= boosted:
            return True
    return False


def _opp_has_stage2(state: dict) -> bool:
    db = get_db()
    op = opp_state(state)
    for p in [active_of(op)] + list(op.get("bench") or []):
        if not p:
            continue
        try:
            if getattr(db.card(p.get("id")), "stage2", False):
                return True
        except Exception:
            pass
    return False


def _lucario_play_score(cid: int, state: dict, hand_n: int, bench_n: int):
    """Bespoke play priorities for the Mega Lucario engine. Returns None to fall through."""
    act = active_of(my_state(state))
    lucario_active = bool(act and act.get("id") == _MEGA_LUCARIO_ID)
    if cid == _PREMIUM_POWER_PRO_ID:
        # One-shot +30 this turn: play it only to convert an attack into a KO.
        return 16.0 if _premium_secures_ko(state) else (3.0 if lucario_active else 1.0)
    if cid == _FIGHTING_GONG_ID:
        return 9.0                                  # search {F} energy / basic — consistency
    if cid == _DUSK_BALL_ID:
        return 8.0                                  # dig a Pokémon
    if cid == _CARMINE_ID:
        fp = state.get("firstPlayer", -1)
        going_first = (fp is not None and fp >= 0 and fp == state.get("yourIndex"))
        if going_first and state.get("turn", 99) <= 2:
            return 15.0                             # strong T1 going-first dig (draw 5)
        return 9.0 + max(0, 6 - hand_n)
    if cid == _GRAVITY_MOUNTAIN_ID:
        return 11.0 if _opp_has_stage2(state) else 5.0  # shrink opp Stage 2s (Dragapult/Walrein)
    return None


def _opp_uses_effect_damage(state: dict) -> bool:
    """True if opponent's active Pokémon's best attack deals 0 direct damage (effect-based).

    Cards like Alakazam (Powerful Hand: place damage counters) or Strange Hacking
    (move counters) don't count as 'damage from attack', so Horrifying Revenge never
    triggers. Mist Energy on our active blocks these effects entirely.
    """
    db = get_db()
    op = opp_state(state)
    op_act = active_of(op)
    if op_act is None:
        return False
    cid = op_act.get("id")
    if cid is None:
        return False
    attacks = db.attacks_of(cid)
    if not attacks:
        return False
    # If ALL attacks do 0 damage the opponent is pure-effect — Mist Energy blocks them.
    return all(a.damage == 0 for a in attacks)


def _score_play(o: dict, state: dict) -> float:
    db = get_db()
    mp = my_state(state)
    op = opp_state(state)
    cid = _card_id_from_option(o, state)
    hand_n = mp.get("handCount", 0)
    bench_n = len(mp.get("bench") or [])
    if cid is None:
        return 5.0
    ct = db.card_type(cid)
    if ct is None:
        return 5.0
    name = ct.name
    # Mega Starmie engine: bespoke trainer priorities (guarded so Hops/Dragapult/Slowking
    # decks, which share some of these IDs, keep their generic scoring).
    if _starmie_in_play(state):
        s = _starmie_play_score(cid, state, hand_n, bench_n)
        if s is not None:
            return s
    if _lucario_in_play(state):
        s = _lucario_play_score(cid, state, hand_n, bench_n)
        if s is not None:
            return s
    if name == "POKEMON":
        if cid == _LILLIE_CLEFAIRY_EX_ID:
            # NEVER play Clefairy unless the opponent has a Dragon-type in play — her ONLY
            # value is Fairy Zone (Dragon → Psychic weakness). Vs a non-Dragon opponent she
            # is dead weight: 190 HP 2-prize bait with no attack. Hard-block her.
            if _opp_active_is_dragon(state):
                return 16.0  # Dragon active now: bench immediately, Fairy Zone live
            if _opp_has_dragon_threat(state):
                return 8.0   # Dragon line on bench: prepare before it comes active
            return -500.0    # No Dragon anywhere: never put her down
        # Meowth ex: bench it only for a reason (Supporter tutor), Pokegear-first. See helper.
        if cid == _MEOWTH_EX_ID:
            return _meowth_bench_score(state, bench_n)
        return 12.0 - 2.0 * bench_n if bench_n < 5 else -1.0
    if name == "SUPPORTER":
        # Boss's Orders: play it only when it gusts up a benched target we can KO this turn.
        if cid == _BOSS_ORDERS_ID:
            return _score_boss_orders(state)
        # Lillie's Determination / Judge / Harlequin: play AFTER items (they shuffle your hand).
        if cid in _HAND_SHUFFLE_DRAWS:
            return _score_hand_shuffle_draw(state)
        if cid == _XEROSIC_ID:
            op_hand = op.get("handCount", 0)
            if op_hand > 3:
                return 12.0 + (op_hand - 3) * 1.5
            return 4.0
        # Brock's Scouting: 2 basics OR 1 evolution — great for finding Trevenant directly
        # or filling bench with Phantump + Snorlax/Cramorant in one play.
        if cid == _BROCKS_SCOUTING_ID:
            bench_n_now = len(mp.get("bench") or [])
            return 10.0 + max(0, 4 - bench_n_now)  # higher priority when bench is thin
        # Ciphermaniac's Codebreaking: search 2, put on top — sets up Seek Inspiration.
        if cid == _CIPHERMANIAC_ID:
            if _slowking_active_charged(state):
                return 15.0  # combo is live this turn: dig + stack the payoff on top
            if _slowking_in_play(state):
                return 11.0
            return 9.0
        return 9.0 + max(0, 6 - hand_n)
    if name == "ITEM":
        if cid == _POKE_PAD_ID:
            return _score_poke_pad(state)
        # Secret Box: fetch Item+Tool+Supporter+Stadium in one card — huge tempo swing.
        # Cost is discarding 3 cards, so only worth it when hand has enough to spare.
        if cid == _SECRET_BOX_ID:
            if hand_n >= 5:
                return 13.0  # big swing: assembles full combo (Band+Postwick+Supporter)
            if hand_n >= 4:
                return 9.0   # tight but can afford the 3 discards
            return 2.0       # can't use it
        # Pokegear 3.0: dig top 7 for a Supporter. PRIORITIZE it when bricked (no Supporter in
        # hand) so it goes BEFORE Meowth ex (the user's rule: Pokegear first, Meowth only if it
        # whiffs). Lower when we already hold a Supporter.
        if cid == _POKEGEAR_ID:
            return 14.0 if not _hand_has_supporter(state) else 6.0
        # Team Rocket's Transceiver: searches any Team Rocket's Supporter (Petrel).
        # More valuable when hand is thin (needs draw engine).
        if cid == _TR_TRANSCEIVER_ID:
            return 9.0 + max(0, 5 - hand_n)
        return 7.0
    if name == "TOOL":
        return 6.0
    if name == "STADIUM":
        # Postwick: override opponent's Watchtower (which shuts our Dudunsparce draw engine)
        if cid == _POSTWICK_ID:
            stadium = state.get("stadium") or []
            watchtower_up = any((c or {}).get("id") == _WATCHTOWER_ID for c in stadium)
            return 12.0 if watchtower_up else 6.0
        # Academy at Night: combo enabler — put a payoff Pokémon on top for Seek Inspiration.
        if cid == _ACADEMY_NIGHT_ID:
            return 13.0 if _slowking_active_charged(state) else 6.5
        return 5.5
    # Special energy: Mist Energy gets high priority when opponent uses effect-based damage
    # (e.g. Alakazam Powerful Hand — places damage counters, not attack damage).
    if cid == _MIST_ENERGY_ID:
        if _opp_uses_effect_damage(state):
            return 14.0  # urgent: attach to block opponent's ability/effect strategy
        return 6.0  # otherwise treat like any energy attachment
    return 5.0


def _attack_energy_need(pokemon: dict, db) -> int:
    """How many more energy the Pokémon needs to fire its cheapest attack."""
    cid = pokemon.get("id")
    if cid is None:
        return 3
    attacks = db.attacks_of(cid)
    if not attacks:
        return 3
    min_cost = min(ai.cost for ai in attacks)
    reduction = _tool_cost_reduction(pokemon)
    return max(0, min_cost - reduction - total_energy(pokemon))


def _score_attach(o: dict, state: dict) -> float:
    """Energy attach: prefer loading the active attacker that still needs energy."""
    db = get_db()
    mp = my_state(state)
    in_play_area = o.get("inPlayArea")
    in_play_idx = o.get("inPlayIndex")

    def _attach_target():
        if in_play_area == 4:
            return active_of(mp)
        if in_play_area == 5:
            bench = mp.get("bench") or []
            if in_play_idx is not None and 0 <= in_play_idx < len(bench):
                return bench[in_play_idx]
        return None

    # ---- Hero's Cape (+100 HP): only ever wants a Mega Starmie, else a Staryu (future Starmie).
    if _card_id_from_option(o, state) == _HEROS_CAPE_ID:
        tgt = _attach_target()
        tid = (tgt or {}).get("id")
        if tid in (_MEGA_STARMIE_ID, _MEGA_LUCARIO_ID):
            return 30.0   # +100 HP on the main attacker wall (330->430 / 340->440)
        if tid in (_STARYU_ID, _RIOLU_ID):
            return 18.0   # or its pre-evolution (future main attacker)
        return -5.0  # never cape anything else

    # ---- Hop's Choice Band (-1 attack cost, +30 dmg): ONLY on a Hop's Pokémon. PRIORITY is
    # always on Phantump / Cramorant when one of them is the Active (attacking this turn) — the
    # -1 cost lets them attack for free and +30 boosts the hit. Trevenant still gets it when it's
    # the active finisher (e.g. the +30 modifier that KOs Mega Lucario). Never on anything else.
    if _card_id_from_option(o, state) == _HOPS_CHOICE_BAND_ID:
        tgt = _attach_target()
        tid = (tgt or {}).get("id")
        is_active = (in_play_area == 4)
        if tid in (_HOPS_PHANTUMP_ID, _HOPS_CRAMORANT_ID):
            return 32.0 if is_active else 24.0   # priority: attacking THIS turn
        if tid == _HOPS_TREVENANT_ID:
            return 28.0 if is_active else 22.0
        # HARD block on any non-attacking-Hop's target (Meowth ex, Dunsparce line, Snorlax,
        # Clefairy...). Must be -500 not -5: the MCTS search treats the score as a prior and a
        # mere -5 still got Choice Band onto Meowth/Dunsparce on the ladder. -500 makes the
        # search prior ~0 so it is effectively never chosen.
        return -500.0

    # ---- NEVER energize the Dunsparce draw line. It must never attack and must never carry
    # energy — not even to pay a retreat. (The never-promote rules keep it off the Active.)
    _atgt = _attach_target()
    if (_atgt or {}).get("id") in (_DUNSPARCE_ID, _DUDUNSPARCE_ID):
        return -50.0

    energy_cid = _card_id_from_option(o, state)

    # ---- Telepath Psychic Energy: on a PSYCHIC Pokémon it searches a Psychic Basic to the
    # bench (free board development) and powers our Psychic attackers (Phantump/Trevenant).
    # Strongly prefer a Psychic target so the search triggers; near-worthless elsewhere.
    if energy_cid == _TELEPATH_PSYCHIC_ENERGY_ID:
        _t = _attach_target()
        try:
            is_psychic = bool(_t and db.card(_t.get("id")).energyType == _PSYCHIC_ENERGY_TYPE)
        except Exception:
            is_psychic = False
        if is_psychic:
            need = _attack_energy_need(_t, db)
            return 26.0 + need * 2.0   # development search + charges a Psychic attacker
        return 4.0                     # no search, just a stranded energy — avoid

    # ---- Legacy Energy: the Pokémon it's on takes +1 Prize when it KOs. It's a PRIZE-RACE
    # finisher — only worth it on a real attacker (Phantump/Trevenant/Cramorant) once the
    # opponent is at 3-4 prizes (closing the game). Otherwise hold it.
    if energy_cid == _LEGACY_ENERGY_ID:
        _t = _attach_target()
        tid = (_t or {}).get("id")
        is_attacker = tid in (_HOPS_PHANTUMP_ID, _HOPS_TREVENANT_ID, _HOPS_CRAMORANT_ID)
        if is_attacker and _opp_prizes(state) in (3, 4):
            return 28.0   # extra prize on KO wins the race — the moment to use it
        if is_attacker:
            return 4.0    # right target, wrong time — prefer other energy, save Legacy
        return 1.0        # never waste the prize-race finisher on a non-attacker

    # ---- Ignition Energy: only worth attaching to an EVOLUTION Pokémon (3 energy).
    # On a Basic it provides 1 energy and is discarded end of turn = wasted. ----
    if energy_cid == _IGNITION_ENERGY_ID:
        target = None
        if in_play_area == 4:
            target = active_of(mp)
        elif in_play_area == 5:
            bench = mp.get("bench") or []
            if in_play_idx is not None and 0 <= in_play_idx < len(bench):
                target = bench[in_play_idx]
        if target is not None:
            try:
                tc = db.card(target.get("id"))
                is_basic = bool(tc and getattr(tc, "basic", False))
            except Exception:
                is_basic = False
            if is_basic:
                return -5.0   # wasted on a Basic (Staryu) — never do this
            need = _attack_energy_need(target, db)
            return 16.0 + need * 2.0  # great on Mega Starmie/Cinderace: instant 3 energy

    # AreaType ACTIVE=4, BENCH=5
    if in_play_area == 4:
        act = active_of(mp)
        if act is not None:
            need = _attack_energy_need(act, db)
            return 14.0 + need * 2.0
        return 14.0
    if in_play_area == 5:
        bench = mp.get("bench") or []
        if in_play_idx is not None and 0 <= in_play_idx < len(bench) and bench[in_play_idx]:
            b_poke = bench[in_play_idx]
            need = _attack_energy_need(b_poke, db)
            # Score bench charging below active, but higher when almost ready
            return 7.0 + max(0, 2 - need) * 1.0
        return 8.0
    return 7.0


def _score_retreat(state: dict) -> float:
    """Retreat: urgent when active is near death and bench has a ready attacker."""
    db = get_db()
    mp = my_state(state)
    act = active_of(mp)
    if act is None:
        return 2.0
    curr_hp = act.get("hp", 100)
    max_hp = act.get("maxHp", curr_hp) or curr_hp
    # Check bench for any energy-ready attacker
    bench = mp.get("bench") or []
    bench_ready = False
    for b in bench:
        if b is None:
            continue
        if _attack_energy_need(b, db) == 0:
            bench_ready = True
            break
    # Cramorant stuck active outside its 3-4 prize window does nothing — swap it out for a
    # real attacker rather than sit there (or fire a 0-damage Fickle Spitting).
    if act.get("id") == _HOPS_CRAMORANT_ID and _opp_prizes(state) not in (3, 4):
        other_bencher = any(b and (b or {}).get("id") not in (None, _DUDUNSPARCE_ID)
                            for b in bench)
        if other_bencher:
            return 8.0

    # Dunsparce/Dudunsparce stuck Active can't attack — retreat to a real Hop's attacker.
    if act.get("id") in (_DUNSPARCE_ID, _DUDUNSPARCE_ID):
        if any(b and (b or {}).get("id") in (_HOPS_PHANTUMP_ID, _HOPS_TREVENANT_ID, _HOPS_CRAMORANT_ID)
               for b in bench):
            return 9.0

    # Mega Starmie PROMOTION: the #1 pilots (Yushin Ito 69-10, keidroid) transition off
    # Cinderace the moment a Mega Starmie is charged — Cinderace is the accel engine, not the
    # attacker. If the active is NOT the Starmie but a charged Mega Starmie sits on the bench,
    # retreat to promote it so it can Nebula/Jetting THIS turn. This is the move our raw
    # heuristic was missing (it sat on Cinderace Turbo-Flaring instead of swinging Starmie).
    if act.get("id") != _MEGA_STARMIE_ID:
        for b in bench:
            if b and b.get("id") == _MEGA_STARMIE_ID:
                e = total_energy(b)
                if e >= 3:
                    return 26.0   # Nebula online — bring the workhorse in now
                if e >= 1:
                    return 16.0   # Jetting-ready — still worth promoting for tempo
    if curr_hp <= 30:
        return 7.0 if bench else 1.0   # very low HP: retreat if possible
    if max_hp > 0 and curr_hp < max_hp * 0.4 and bench_ready:
        return 5.0  # moderately damaged with a ready bencher — worth switching
    return 2.0  # default: stay and attack


def _score_ability(o: dict, state: dict) -> float:
    """Score using an ability. Handles Dudunsparce 'Run Away Draw' explicitly."""
    mp = my_state(state)
    # Identify which Pokémon's ability this is. ABILITY options carry the in-play location in
    # `area`/`index` (4=ACTIVE, 5=BENCH) — NOT inPlayArea/inPlayIndex (those are always None
    # here, which silently broke every ability-specific score below).
    area = o.get("area")
    idx = o.get("index")
    cid = None
    if area == 4:
        act = active_of(mp)
        cid = act.get("id") if act else None
    elif area == 5:
        bench = mp.get("bench") or []
        if idx is not None and 0 <= idx < len(bench) and bench[idx]:
            cid = bench[idx].get("id")

    # Dudunsparce "Run Away Draw": draw 3, then shuffle Dudunsparce back into the deck.
    # It is FREE — using it does NOT end the turn — so when we want cards we must use it
    # BEFORE attacking, or the attack ends the turn and the draw is wasted. Score it above
    # attacks while the hand is thin; once the hand is healthy, hold it (it shuffles the
    # Dudunsparce away, so don't burn the engine for cards we don't need).
    if cid == _DUDUNSPARCE_ID:
        hand_n = mp.get("handCount", 0)
        return 110.0 if hand_n <= 4 else 8.0

    # Cinderace: use its ability every time it's available (the Turbo Flare / Explosiveness
    # accel engine is the whole deck). Top ability priority.
    if cid == _CINDERACE_ID:
        return 30.0

    # Lunatone "Lunar Cycle": discard a {F} energy to draw 3 (engine only offers it when
    # Solrock is in play + a Fire energy is discardable). Strong refill — scale with hand.
    if cid == _LUNATONE_ID:
        hand_n = mp.get("handCount", 0)
        if hand_n <= 2:
            return 16.0
        if hand_n <= 4:
            return 12.0
        return 8.0

    # Snorlax "Extra Helpings" is passive — it shouldn't appear as an activatable option.
    # All other abilities: default high priority.
    return 10.0


def _score_main_option(o: dict, state: dict) -> float:
    t = o.get("type")
    if t == OptionType.ATTACK.value:
        return _score_attack(o, state)
    if t == OptionType.PLAY.value:
        return _score_play(o, state)
    if t == OptionType.ATTACH.value:
        return _score_attach(o, state)
    if t == OptionType.ABILITY.value:
        return _score_ability(o, state)
    if t == OptionType.EVOLVE.value:
        return 11.0
    if t == OptionType.RETREAT.value:
        return _score_retreat(state)
    if t == OptionType.DISCARD.value:
        return 1.0
    if t == OptionType.END.value:
        return 3.0  # baseline: act if anything useful exists, else end
    return 4.0


def _score_yesno(o: dict, ctx: int) -> float:
    is_yes = (o.get("type") == OptionType.YES.value)
    # default YES for beneficial activations; specific contexts overridden below
    if ctx == SelectContext.MULLIGAN.value:
        # Always keep (NO). For the Starmie deck a Cinderace hand is keepable via its
        # Explosiveness ability even with no Staryu — never throw it away here.
        return 1.0 if not is_yes else 0.0          # keep hand
    if ctx == SelectContext.IS_FIRST.value:
        return 1.0 if is_yes else 0.0              # go first
    # ACTIVATE / FIRST_EFFECT / COIN_HEAD / others: prefer YES
    return 1.0 if is_yes else 0.0


def _score_opp_energy_target(o: dict, state: dict):
    """When discarding an opponent's energy (Crushing Hammer), strip their MAIN ATTACKER —
    the Pokémon carrying the most energy. Returns a score, or None if not an opp-energy target."""
    pi = o.get("playerIndex")
    if pi is None or pi == state.get("yourIndex"):
        return None
    poke = _get_pokemon_from_option(o, state)
    if poke is None:
        return 10.0
    return 10.0 + total_energy(poke) * 5.0  # more energy on it = higher priority (main attacker)


def _score_card_select(o: dict, ctx: int, state: dict) -> float:
    """Generic card-target scoring for the many CARD contexts."""
    db = get_db()
    opp_e = _score_opp_energy_target(o, state)
    if opp_e is not None and o.get("type") in (OptionType.ENERGY_CARD.value, OptionType.ENERGY.value):
        return opp_e
    cid = _card_id_from_option(o, state)

    # ---- Active/Bench attacker selection (phase-aware for Hops deck) ----
    if ctx in (SelectContext.SETUP_ACTIVE_POKEMON.value, SelectContext.TO_ACTIVE.value,
               SelectContext.SWITCH.value):
        if cid is not None:
            op = opp_state(state)
            op_pr = len(op.get("prize") or [])
            our_kos = 6 - op_pr

            # Dunsparce / Dudunsparce: bench-only draw line. Never promote to Active unless it's
            # the only legal choice (and since we never energize it, a stuck one can't retreat —
            # so it's doubly important to keep it off the Active Spot).
            if cid in (_DUNSPARCE_ID, _DUDUNSPARCE_ID):
                return 1.0

            # Meowth ex: bench-only consistency tutor — 170 HP 2-prize bait if sent active.
            if cid == _MEOWTH_EX_ID:
                return 1.0

            # Lillie's Clefairy ex: bench-ONLY for Fairy Zone ability. 190 HP = 2-prize bait
            # if active. NEVER promote her to the active — only forced if she is the
            # literal last Pokémon in play.
            if cid == _LILLIE_CLEFAIRY_EX_ID:
                return -500.0

            # Snorlax: bench-only role (Extra Helpings ability). Only send active as last resort.
            if cid == _HOPS_SNORLAX_ID:
                bench = my_state(state).get("bench") or []
                has_other_benchers = any(
                    (b or {}).get("id") not in (None, _HOPS_SNORLAX_ID, _DUDUNSPARCE_ID)
                    for b in bench
                )
                return 2.0 if has_other_benchers else 10.0  # last resort only

            opp_lucario = _opp_is_lucario(state)

            # Cramorant: only send active when Fickle Spitting window is open (3-4 prizes).
            # Vs Lucario, additionally require a sniped target it can KO (Solrock / Lunatone /
            # damaged Hariyama) — never promote it just to whiff on the 340 HP Mega.
            if cid == _HOPS_CRAMORANT_ID:
                if op_pr in (3, 4) and (not opp_lucario or _cramorant_target_ok(state)):
                    return 30.0
                return 4.0

            # Phantump: the Lucario gameplan's stall attacker — keep it active vs Lucario until
            # we've taken a KO (then Trevenant takes over). Otherwise early-game protection.
            if cid == _HOPS_PHANTUMP_ID:
                if opp_lucario:
                    return 28.0 if our_kos == 0 else 8.0
                return 22.0 if op_pr == 6 else 10.0

            # Trevenant: priority after we've taken KOs (Horrifying Revenge live) — vs Lucario
            # this is THE finisher (130 + Snorlax/Band/Postwick modifier, 2x Psychic = KOs the
            # 340 HP Mega). Also good mid-game with Corner (90 + up to 90 from modifiers).
            if cid == _HOPS_TREVENANT_ID:
                if opp_lucario and our_kos > 0:
                    return 40.0
                return 26.0 if our_kos > 0 else 18.0

            # Mega Starmie: when promoting (after a KO or via switch), prefer a charged Mega
            # Starmie over Cinderace — Cinderace is accel, Starmie is the attacker. A charged
            # Starmie active = it attacks (Nebula/Jetting) immediately.
            if cid == _MEGA_STARMIE_ID:
                mega = next((p for p in [active_of(my_state(state))] + list(my_state(state).get("bench") or [])
                             if p and p.get("id") == _MEGA_STARMIE_ID), None)
                e = total_energy(mega) if mega else 0
                return 40.0 if e >= 3 else (30.0 if e >= 1 else 12.0)

            # Cinderace: open it (via Explosiveness) — it's the Turbo Flare accel engine that
            # powers up the bench Staryu/Mega Starmie. Prefer it over a passive Staryu, but a
            # charged Mega Starmie (scored above) should come in ahead of it to attack.
            if cid == _CINDERACE_ID:
                return 24.0

            # All other Pokémon: damage-based heuristic
            a = db.best_attack(cid)
            if a:
                return 20.0 - 3.0 * a.cost + a.damage / 50.0
        return 5.0
    if ctx in (SelectContext.SETUP_BENCH_POKEMON.value, SelectContext.TO_BENCH.value,
               SelectContext.TO_FIELD.value):
        # Lillie's Clefairy ex: never field her (even at initial setup) unless the opponent
        # has a Dragon — same rule as playing her from hand. Without this guard the flat 10
        # below would bench her vs Alakazam/Hops where she is dead 2-prize bait.
        if cid == _LILLIE_CLEFAIRY_EX_ID:
            if _opp_active_is_dragon(state):
                return 16.0
            if _opp_has_dragon_threat(state):
                return 8.0
            return -500.0
        # Meowth ex: field it only for the Supporter tutor, Pokegear-first (shared helper).
        if cid == _MEOWTH_EX_ID:
            return _meowth_bench_score(state, len(my_state(state).get("bench") or []))
        return 10.0

    # ---- Damage targeting: KO the lowest-HP opponent Pokémon ----
    if ctx in (SelectContext.DAMAGE_COUNTER.value, SelectContext.DAMAGE_COUNTER_ANY.value,
               SelectContext.DAMAGE.value):
        pi = o.get("playerIndex")
        is_opp = (pi is not None and pi != state.get("yourIndex"))
        poke = _get_pokemon_from_option(o, state)
        if poke is not None:
            hp = poke.get("hp", 9999)
            if is_opp:
                card = db.card(poke.get("id")) if poke.get("id") else None
                prize_mult = 2.0 if (card and card.ex) else 1.0
                return (500.0 / (hp + 1)) * prize_mult
            else:
                return -10.0  # don't damage own Pokémon
        return 5.0

    # ---- Healing: prefer most damaged own Pokémon ----
    if ctx in (SelectContext.REMOVE_DAMAGE_COUNTER.value, SelectContext.HEAL.value):
        poke = _get_pokemon_from_option(o, state)
        if poke is not None:
            hp = poke.get("hp", 0)
            max_hp = poke.get("maxHp", hp) or hp
            damage_taken = max_hp - hp
            return float(damage_taken)  # highest damage = highest priority to heal
        return 5.0

    # ---- Discard: prefer expendable cards, keep Pokémon and key Supporters ----
    if ctx in (SelectContext.DISCARD.value, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD.value):
        if cid is not None:
            if db.is_basic_energy(cid):
                return 10.0  # energy is most expendable
            if db.is_energy(cid):
                return 8.0
            ct = db.card_type(cid)
            if ct is not None:
                name = ct.name
                if name == "ITEM":
                    return 6.0
                if name == "TOOL":
                    return 5.0
                if name == "STADIUM":
                    return 5.0
                if name == "SUPPORTER":
                    return 4.0
                if name == "POKEMON":
                    return 2.0  # preserve Pokémon
        return 5.0

    # ---- Return to deck: prefer putting back basic energy ----
    if ctx in (SelectContext.TO_DECK.value, SelectContext.TO_DECK_BOTTOM.value):
        # Slowking combo: when Seek Inspiration is live this turn, put a payoff Pokémon on
        # TOP (TO_DECK only — never the bottom) so Seek discards and copies its attack.
        if (ctx == SelectContext.TO_DECK.value and cid in _COMBO_PAYOFFS
                and _slowking_active_charged(state)):
            return 42.0 if cid == _best_combo_payoff(state) else 32.0
        if cid is not None:
            if db.is_basic_energy(cid):
                return 10.0
            ct = db.card_type(cid)
            if ct and ct.name == "POKEMON":
                return 2.0
        return 5.0

    return 5.0


def option_scores(obs: dict) -> list[float]:
    sel = obs["select"]
    state = obs["current"]
    ctx = sel["context"]
    opts = sel["option"]
    scores = []
    for o in opts:
        t = o.get("type")
        if t in (OptionType.YES.value, OptionType.NO.value):
            s = _score_yesno(o, ctx)
        elif t in (OptionType.CARD.value, OptionType.TOOL_CARD.value, OptionType.ENERGY_CARD.value):
            s = _score_card_select(o, ctx, state)
        elif t == OptionType.ENERGY.value:
            opp_e = _score_opp_energy_target(o, state)
            s = opp_e if opp_e is not None else 5.0
        elif t == OptionType.NUMBER.value:
            s = float(o.get("number") or 0)
        elif t == OptionType.SPECIAL_CONDITION.value:
            s = 5.0
        elif t == OptionType.SKILL.value:
            s = 5.0
        else:
            s = _score_main_option(o, state)
        scores.append(s)
    return scores


def _choose_count(sel: dict) -> int:
    lo, hi = sel["minCount"], sel["maxCount"]
    n = len(sel["option"])
    hi = min(hi, n)
    if hi <= lo:
        return max(0, lo)
    ctx = sel["context"]
    if ctx in _BENEFICIAL_COUNT:
        return hi
    if ctx in _COSTLY_COUNT:
        return lo
    # default: act minimally but at least once when allowed
    return lo if lo > 0 else 1


def choose(obs: dict, rng: random.Random | None = None, epsilon: float = 0.0) -> list[int]:
    """Greedy (optionally epsilon-noisy) contract-valid selection."""
    sel = obs["select"]
    n = len(sel["option"])
    if n == 0:
        return []
    k = _choose_count(sel)
    if k <= 0:
        return []
    rng = rng or random
    if epsilon > 0 and rng.random() < epsilon:
        return rng.sample(range(n), min(k, n))
    scores = option_scores(obs)
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return sorted(order[:k])

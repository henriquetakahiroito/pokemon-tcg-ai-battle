"""Stress-test the ported Mega Lucario rule agent for consistency.

Runs Lucario vs a gauntlet of opponent decks/agents:
  - Lucario mirror (does it beat itself coherently / no crashes?)
  - Lucario (rule) vs Hops (our MCTS) — cross-archetype
  - Lucario (rule) vs Starmie (our MCTS) — the deck we pilot badly
  - Lucario (rule) vs a few meta decks piloted by our heuristic

Reports win/loss/draw, illegal-action crashes, and avg plies. Consistency = high winrate
with ZERO crashes (a crash on Kaggle = a dead 'Error' submission).
"""
import sys, time, traceback
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from cg.game import battle_start, battle_select
from agent.base import read_deck
from agent.lucario import LucarioAgent
from agent.agent import MctsAgent


def play(deckA, makeA, deckB, makeB, max_plies=900):
    a, b = makeA(), makeB()
    obs, _ = battle_start(deckA, deckB)
    plies = 0
    while obs["current"]["result"] < 0 and obs.get("select") is not None:
        who = obs["current"]["yourIndex"]
        try:
            act = a(obs) if who == 0 else b(obs)
        except Exception:
            traceback.print_exc()
            return None, plies, who  # crash
        obs.pop("search_begin_input", None)
        obs = battle_select(act)
        plies += 1
        if plies > max_plies:
            break
    return obs["current"]["result"], plies, None


def run(label, deckL, deckO, makeO, N=20):
    w = l = d = crash = 0; tot_plies = 0; t0 = time.time()
    for g in range(N):
        if g % 2 == 0:
            r, p, cw = play(deckL, lambda: LucarioAgent(deck=deckL, seed=g),
                            deckO, makeO); lu = 0
        else:
            r, p, cw = play(deckO, makeO,
                            deckL, lambda: LucarioAgent(deck=deckL, seed=g)); lu = 1
        tot_plies += p
        if r is None:
            crash += 1
        elif r == lu:
            w += 1
        elif r == 1 - lu:
            l += 1
        else:
            d += 1
    dt = time.time() - t0
    wr = w / max(w + l, 1) * 100
    print(f"{label:32s} Lucario {w}-{l}-{d}  (winrate {wr:3.0f}%)  "
          f"crashes={crash}  avg_plies={tot_plies/N:.0f}  {dt:.0f}s", flush=True)
    return w, l, d, crash


if __name__ == "__main__":
    LU = read_deck("deck_lucario.csv")
    HOPS = read_deck("deck_cand_hops_v9_clef_meowth.csv")
    STARMIE = read_deck("deck_cand_starmie.csv")
    DRAGA = read_deck("deck_cand_dragapult_official.csv")
    ALAKAZAM = read_deck("deck_meta_alakazam.csv")

    print("=== Mega Lucario (v63 rule agent) stress gauntlet ===\n")
    run("mirror vs self",            LU, LU,       lambda: LucarioAgent(deck=LU, seed=999))
    run("vs Hops (MCTS)",            LU, HOPS,     lambda: MctsAgent(deck=HOPS, seed=777))
    run("vs Starmie (MCTS)",         LU, STARMIE,  lambda: MctsAgent(deck=STARMIE, seed=777))
    run("vs Dragapult (MCTS)",       LU, DRAGA,    lambda: MctsAgent(deck=DRAGA, seed=777))
    run("vs Alakazam ctrl (MCTS)",   LU, ALAKAZAM, lambda: MctsAgent(deck=ALAKAZAM, seed=777))

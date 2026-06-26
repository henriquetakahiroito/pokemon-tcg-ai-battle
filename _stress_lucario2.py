"""Fast Lucario consistency gauntlet — opponents piloted by the greedy heuristic (policy.choose),
which is fast enough to run every matchup. Plus the direct test that matters: the Lucario rule
agent vs the Starmie deck piloted by OUR engine (the deck we score 561 ELO with)."""
import sys, time, random, traceback
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from cg.game import battle_start, battle_select
from agent.base import read_deck, BaseAgent
from agent.lucario import LucarioAgent
from agent.policy import choose as greedy_choose


class GreedyAgent(BaseAgent):
    name = "greedy"
    def decide(self, obs):
        return greedy_choose(obs, rng=self.rng)


def play(deckA, makeA, deckB, makeB, max_plies=900):
    a, b = makeA(), makeB()
    obs, _ = battle_start(deckA, deckB); plies = 0
    while obs["current"]["result"] < 0 and obs.get("select") is not None:
        who = obs["current"]["yourIndex"]
        try:
            act = a(obs) if who == 0 else b(obs)
        except Exception:
            traceback.print_exc(); return None, plies
        obs.pop("search_begin_input", None); obs = battle_select(act); plies += 1
        if plies > max_plies: break
    return obs["current"]["result"], plies


def run(label, deckL, deckO, makeO, N=30):
    w = l = d = crash = 0; tp = 0; t0 = time.time()
    for g in range(N):
        if g % 2 == 0:
            r, p = play(deckL, lambda: LucarioAgent(deck=deckL, seed=g), deckO, makeO); lu = 0
        else:
            r, p = play(deckO, makeO, deckL, lambda: LucarioAgent(deck=deckL, seed=g)); lu = 1
        tp += p
        if r is None: crash += 1
        elif r == lu: w += 1
        elif r == 1 - lu: l += 1
        else: d += 1
    wr = w / max(w + l, 1) * 100
    print(f"{label:34s} Lucario {w:2d}-{l:2d}-{d}  (wr {wr:3.0f}%)  crashes={crash}  "
          f"plies={tp/N:.0f}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    LU = read_deck("deck_lucario.csv")
    decks = {
        "Hops": "deck_cand_hops_v9_clef_meowth.csv",
        "Starmie": "deck_cand_starmie.csv",
        "Starmie (Jaga)": "deck_cand_starmie_jaga.csv",
        "Dragapult": "deck_cand_dragapult_official.csv",
        "Dragapult (Munki)": "deck_cand_dragapult_munkidori.csv",
        "Alakazam ctrl": "deck_meta_alakazam.csv",
        "Crustle wall": "deck_crustle.csv",
        "Abomasnow water": "deck_meta_abomasnow.csv",
        "Slowking combo": "deck_cand_slowking_combo.csv",
        "Walrein stall": "deck_cand_walrein.csv",
        "Sylveon": "deck_cand_sylveon.csv",
        "Tea Party": "deck_cand_teaparty.csv",
        "Dark Yveltal": "deck_cand_dark_yveltal.csv",
        "Fight Koraidon": "deck_cand_fight_koraidon.csv",
        "Rocket Spidops": "deck_cand_tr_spidops.csv",
        "Lucario mirror": "deck_lucario.csv",
        "Fire ex": "deck_meta_fire_ex.csv",
        "Non-ex": "deck_meta_nonex.csv",
    }
    print("=== Lucario rule agent vs greedy-piloted gauntlet ===\n")
    for name, path in decks.items():
        try:
            od = read_deck(path)
        except Exception as e:
            print(f"  skip {name}: {e}"); continue
        run(f"vs {name}", LU, od, lambda od=od: GreedyAgent(deck=od, seed=777))

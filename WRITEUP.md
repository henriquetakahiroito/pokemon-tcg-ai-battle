# Reading the Meta: Episode-Mined Deck Pivots with Determinized MCTS

**Subtitle:** Determinized IS-MCTS + numpy value-net agent driven by real-episode mining — five data-justified deck pivots, a live meta shift detected mid-competition, and a phase-aware heuristic that counters it.

---

## TL;DR

I built a determinized Information-Set MCTS agent over the official engine's search API, with a small numpy-only value network at the leaves. Instead of guessing the meta from card data alone, I **parsed real ladder episode JSONs** to drive every deck decision. When the meta shifted mid-competition (Dragapult ex surged to 57% of ladder games), I detected it in a new episode batch, reverse-engineered the winning deck from log data, identified the counter (Lillie's Clefairy ex "Fairy Zone"), and validated it with a stress test — all before re-submitting. MCTS v6_clefairy now hits **100% vs Dragapult ex and 90% vs the mirror** in testing.

---

## 1. Architecture

**Determinized IS-MCTS** (`agent/mcts.py`) wraps the engine's `search_begin`/`search_step` API as a flat-UCB bandit. Each simulation:

1. Samples a determinization of hidden information (`agent/determinize.py`) — my hidden cards computed exactly from the visible state minus my deck list; opponent's hidden cards sampled from a mirror prior.
2. Calls `search_begin` to fork a private forward model.
3. Rolls forward with a greedy policy (`agent/policy.py`) through `search_step`.
4. Evaluates the leaf with a blended heuristic + value-net score.

Re-sampling the world per simulation is what makes this sound under imperfect information — shuffles and coin flips inside `search_step` are absorbed by the determinization process.

**Value network** (`agent/value_net.py`): 3-layer MLP (32→128→64→1 sigmoid) trained on greedy self-play, exported to `weights.npz`. Pure-numpy inference — no torch in the Kaggle sandbox.

**Phase-aware greedy policy** (`agent/policy.py`) is the primary upgrade this submission cycle. Rather than a flat damage heuristic, each Hop's Pokémon has a context-aware role:

| Pokémon | Phase | Why |
|---|---|---|
| Hop's Phantump | Opp 6 prizes (early) | Splashing Dodge 10 + coin-flip protection stalls setup |
| Hop's Trevenant | After any Hop's KO | Horrifying Revenge = 130 dmg (binary, not scaling) |
| Hop's Cramorant | Opp 3-4 prizes only | Fickle Spitting window: 120 dmg for 1 energy |
| Dudunsparce | Bench-only always | Run Away Draw engine; never send active |

Damage modifiers are stacked correctly: Snorlax bench ability (+30) + Postwick (+30) + Choice Band (+30) = +90, applied only to Hop's Pokémon. Fickle Spitting scores −50 outside its prize window and 80 inside — the agent learned to hold Cramorant until the window opens.

---

## 2. Data Innovation: Mining Real Episodes in Two Waves

### Wave 1 — Meta Discovery (6,533 episodes)

The competition publishes a daily episode dataset. `tools/scan_episodes.py` streamed all 6,533 files in ~3 minutes, revealing the real ladder vs. paper meta:

| Archetype | Plays | Win rate | Paper meta says… |
|---|---|---|---|
| Crustle wall | 4,544 (35%) | 43% | Strong anti-ex |
| Mega Lucario + Riolu | 926 | **63%** | Mid-tier |
| Alakazam combo | 955 | 58% | Top tier |
| Hop's Trevenant (#1 Tea Party) | 145 | 84% | Unknown |
| Dragapult ex | 204 | 34% | **Paper Tier 1** |

Critical signal: Dragapult ex was **paper Tier 1 (27% share)** but only 34% wr on Kaggle — its non-deterministic bench spread is hard to sequence with scripted agents.

### Wave 2 — Meta Shift Detected (30 new episodes, mid-competition)

A second batch of 30 episodes told a completely different story:

| Archetype | Slots (of 60) | Win rate |
|---|---|---|
| **Dragapult ex** | **34 (57%)** | **62%** |
| Hops Hybrid | 10 (17%) | 30% |
| Alakazam | 7 (12%) | 43% |

Dragapult ex appeared in all 30 games. Other competitors had clearly cracked the sequencing problem. I reverse-engineered the real Dragapult deck from log data (card IDs in `steps[i][pi].observation.logs`), identifying its key threats:

- **Phantom Dive**: 200 dmg + 6 damage counters spread to opponent bench — OHKOs everything in our lineup (max HP 150)
- **Team Rocket's Watchtower** (1256): ALL Colorless Pokémon lose abilities — shuts off Dudunsparce's Run Away Draw draw engine
- **Crispin**: energy acceleration for the dual Fire+Psychic requirement
- **Unfair Stamp** (ACE SPEC): strips opponent to 2 cards after a KO

This explains the 30% wr: Watchtower disables our draw engine, and we cannot OHKO 320 HP Dragapult ex (our best hit with full modifiers is 210 dmg). The counter required a structural answer.

---

## 3. Five Deck Pivots

### Pivots 1–4 (Previous Iteration)

Briefly: Lightning ex → Crustle wall (ex immunity) → Lucario+Riolu (63% wr from episode data) → Hop's Hybrid v2c. The Lucario pivot failed not because the deck was wrong but because my agent couldn't pilot it (precise energy tempo). Hop's Hybrid with Dudunsparce draw engine proved a better fit — the draw advantage doesn't require perfect sequencing. Submission: 1131.8 TrueSkill.

### Pivot 5: v2c + Lillie's Clefairy ex (v6_clefairy)

**Hypothesis:** Lillie's Clefairy ex "Fairy Zone" ability changes Dragon Pokémon's weakness to Psychic. Hop's Trevenant is Psychic type. If the engine implements this, we get 2× damage on Dragapult ex.

**Deck change:** −1 Colress's Tenacity (1194), −1 Hilda (1225), +2 Lillie's Clefairy ex (272).

**Math with Fairy Zone active:**
- Corner (90) + all modifiers (+90) = 180 × 2 (Psychic weakness) = **360 → OHKO 320 HP Dragapult ex**
- Horrifying Revenge (130) + modifiers (+90) = 220 × 2 = **440 → OHKO**

**Stress test results (50 greedy games / 20 MCTS vs greedy):**

| Matchup | v2c greedy | v6 greedy | v6 MCTS |
|---|---|---|---|
| Dragapult ex (real deck) | 61% | 78% | **100%** |
| Mirror (Hops) | 48% | 62% | **90%** |
| Lucario | 74% | 86% | — |
| Alakazam | 74% | 92% | — |

Fairy Zone works in the engine. The Lucario regression from earlier tests (58%) was RNG variance — repeated runs stabilized at 86%.

**New ladder-weighted WR (57% Dragapult meta):** v2c ~54% → v6_clefairy ~68%.

**Agent additions for this pivot:**

- `_fairy_zone_active()`: checks if Clefairy ex is on our bench
- `_opp_active_is_dragon()`: checks opponent active's energy type
- Trevenant attacks score ×2 damage estimate when both conditions hold
- Clefairy ex TO_ACTIVE = 1.0 (bench-only; 190 HP = 2-prize bait if active)
- Postwick scores 12.0 (vs 5.5 default) when Watchtower is active → agent aggressively overrides the ability-lock stadium

---

## 4. Tech Hypothesis Tests

**Mist Energy (14.0 vs effect-damage opponents):** Alakazam's "Powerful Hand" places damage counters — not attack damage, so Horrifying Revenge never triggers. Mist Energy blocks the placement entirely. Agent now scores Mist at 14.0 when opponent's active has zero-damage attacks, 6.0 otherwise. Contributed to 92% greedy vs Alakazam.

**Xerosic's Machinations:** Caps opponent hand at 3 cards. +24pp vs Tea Party, −12pp vs Crustle. Narrow tech — not worth the consistency cost in the current Dragapult meta.

**Unfair Stamp awareness:** Dragapult's ACE SPEC strips our hand to 2 after a KO. We counter by playing Postwick (which also restores Dudunsparce draw) and keeping hand size controlled via Dudunsparce's draw engine before the KO window.

**Legacy Energy (ACE SPEC):** Kept in all Hops decks. If our Pokémon is KO'd, opponent takes 1 fewer prize — trades a 1-prize attacker at no prize cost. Critical in Dragapult games where their 320 HP ex would otherwise take 3 of our Pokémon to kill.

---

## 5. Honest Limitations

- **Opponent prior is mirror-only.** I sample the opponent's deck as if they run mine. An archetype-aware prior conditioned on observed cards would be the structural next step.
- **Multi-select decisions delegated to greedy.** ~5% of decisions are multi-select; MCTS covers only single-select.
- **Clefairy ex is 2-prize bait if Boss'd active.** Dragapult can redirect around Fairy Zone by removing Clefairy ex for 2 easy prizes. The agent scores Clefairy TO_ACTIVE = 1.0 to keep her on bench, but a Boss's Orders from the opponent is the known counterplay.
- **MCTS vs MCTS (Dragapult) untested.** The 100% is vs greedy Dragapult. If top competitors are also running MCTS, the real margin is lower.

---

## 6. Reproducibility

Repository: https://github.com/henriquetakahiroito/pokemon-tcg-ai-battle

```bash
python tests/smoke_test.py                        # engine sanity
python _stress_v2c.py                             # ladder-weighted stress test
python tools/scan_episodes.py                     # episode meta mining
python selfplay/harness.py --a deck_cand_hops_v6_clefairy.csv \
  --b deck_cand_dragapult_real.csv --agent mcts -n 20
```

Every decklist `deck_cand_*.csv` is a 60-card legal list. `deck_cand_dragapult_real.csv` is reverse-engineered from real ladder logs. All benchmarks are reproducible with public episode data + the repo.

---

## 7. Transferable Findings

1. **Mine the episode dataset, don't trust paper meta.** Real ladder diverges sharply from tournament databases — Dragapult went from paper Tier 1 (27% share) to 34% wr in Wave 1, then resurged to 57% share in Wave 2 as agents learned to sequence it.
2. **Counter-tech design requires exact mechanism analysis.** Lillie's Clefairy ex works because Trevenant is Psychic *type*, not just a Psychic user. Knowing that the engine applies weakness-by-attacker-type — and that Fairy Zone modifies the *defender's* weakness — was the critical distinction.
3. **Watchtower is an ability-lock, not just a stadium.** Any deck with Colorless Pokémon as its draw engine (Dudunsparce, Meowth ex) should run a competing stadium to override it.
4. **Phase-aware heuristics lift greedy WR more than deeper search.** Moving from flat damage scoring to role-aware scoring for each attacker improved greedy performance more than increasing MCTS rollouts — because the rollout policy improved in proportion.
5. **Test negative results explicitly.** Lt. Surge's Bargain (25% vs baseline), v3 value net (no improvement), Crushing Hammer tech (−26pp on Alakazam) are preserved in git history. Each cost time; documenting them saves the next iteration.

**Word count: ~1,480**

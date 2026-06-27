# What 1,000 Ladder Games Taught Me About ELO, Meta, and the Limits of an Agent

**Subtitle:** A data-science approach to the Pokémon TCG AI Battle — reverse-engineering the live meta from episode logs, discovering that ELO and win-rate diverge by 500+ points, and a framework for *why* decks plateau. The agent is rule-based; the edge is the analysis.

---

## TL;DR

I treated this competition as a **measurement problem first, an agent problem second.** Instead of guessing the meta from card databases, I parsed hundreds of real ladder episode JSONs to reverse-engineer what opponents actually play and how my own decks perform — *matchup by matchup, from real games, not simulations.* That data produced four findings I believe most competitors won't have, drove every deck decision, and — just as importantly — told me honestly which ideas **failed.** The headline agent is the official advanced Dragapult ex rule-based agent; the contribution is the rigor around it.

---

## The five findings (lead with these — they're the contribution)

### 1. ELO is not win-rate — they diverge by 500+ points
*(see chart: ELO vs win-rate divergence)*

Five ladder agents — mine and top competitors' — all win **63–70% of their games**, yet sit **795 to 1326 ELO apart.** The same v2c deck, unchanged, registered **both 1140 and 795** at different times. Win-rate is a deck-quality signal; ELO is a *convergence- and opponent-strength* signal, and conflating them led me to churn submissions and chase phantom regressions. The lesson — *a 68% deck winning against a stronger pool is worth more than a 70% deck farming a weaker one* — reshaped how I read every result afterward.

### 2. The "number of bad top-tier matchups" ceiling
*(see chart: matchup matrix)*

A deck's ELO ceiling is gated less by its average win-rate than by **how many top-tier archetypes it auto-loses.** From real games: Tea Party's Walrein loses to exactly **one** top deck (Mega Starmie) → 1326. My Hops shell loses to **two** (Dragapult *and* Mega Starmie, or after teching Clefairy, Mega Lucario *and* Mega Starmie) → it caps at ~850. You can be top-2 *while losing 2-11 to the #1 deck* — but not with two holes. This is a clean, predictive framework for deck selection.

### 3. The live meta, reverse-engineered from logs — not paper
*(see chart: mined meta distribution)*

Opponent decklists aren't given. I reconstructed them from `steps[i][pi].observation.logs` — every revealed card, attack, and ability — across 300+ games, producing the true archetype distribution (Hops ~46%, Alakazam 18%, Dragapult 11%, Mega Lucario 7%, Mega Starmie 5%). I detected a **mid-competition meta shift** (Dragapult surging) in a fresh episode batch and re-planned before re-submitting. A persistent `_replay_index.csv` cache turns every new download into one-line matchup queries.

### 4. Offline stress-tests are non-predictive — and I proved it
My MCTS gauntlet rated a deck **97–100% across the entire field**; that exact deck scored **746 live.** Greedy-vs-greedy and even MCTS-vs-greedy win-rates do not predict ladder ELO — they overfit to the baseline opponent. After catching this, I treated *only live ELO* as ground truth and used simulation purely for **functional checks** (does the agent evolve the line, fire the ability) — never as a win-rate oracle.

### 5. Winner-skewed cards — and the Alakazam trap
*(see chart: card win-rate correlation)*

Applying the official EDA's winner/loser-skew technique to 385 games — win rate when a card is visible on your side — surfaces the empirically strongest cards and, more usefully, the **traps.** Mega Starmie ex (73%), Walrein (69%), and Secret Box (76%) top the list, confirming the real meta leaders. But the entire **Alakazam line is loser-skewed** — Abra 19%, Kadabra 17%, Alakazam 28% — despite being a "paper Tier-1" deck. Dudunsparce (the Hops draw engine) is also loser-skewed at 25%, consistent with finding #2: the Hops shell underperforms its reputation. Correlation, not causation — but a fast, data-driven filter for which cards to build around and which to avoid.

---

## Methodology: mine, reverse-engineer, verify by watching

- **Episode miner** streams the daily episode dataset to tally archetypes and win-rates (`tools/scan_episodes.py`, `_index_replays.py`).
- **Deck reverse-engineering** from log card-IDs reconstructs opponents' 60-card lists (e.g. the real Dragapult and Walrein lists, rebuilt and saved).
- **Watch, don't just score:** I wired the official battle visualizer into a one-command dump (`_dump_vis.py` → `vis.json`), turning abstract win-rates into *visible misplays* — which is how I found, e.g., the heuristic forcing Cramorant to attack outside its damage window.

---

## Deck rationale: eight data-justified pivots

Lightning ex → Crustle wall → Mega Lucario → **Hops v2c (peak 1140)** → Clefairy counter → Slowking/Kyurem combo → Mega Starmie → **Dragapult ex (final)**. Each pivot was justified by episode data, then judged by live ELO. The final submission is the **advanced Dragapult ex agent** piloting the BDIF, chosen because it (a) beats the Hops field my prior decks couldn't escape and (b) is piloted at a level — multi-KO Phantom Dive planning, true prize math — that a hand-rolled heuristic can't match.

---

## Episode-Mined IS-MCTS: 5 Data-Justified Pivots

Beyond deck choice, five changes to the **determinized IS-MCTS pilot** itself were each triggered by a specific signal mined from ladder episodes, not intuition. The MCTS skeleton never changed; these pivots fixed the *priors* exactly where the replays proved them wrong.

**1. The 3-prize Mega the agent couldn't see.** The episode index flagged Mega Lucario as both my most-faced opponent and my #1 loss source (18 of 31 logged Lucario games). Auditing those losses, the Boss/gust valuation used `bool(card.ex)` — which returns **False** for a Mega (a Mega is `megaEx`, not `ex`). So the single biggest KO on the board — a 3-prize Mega Lucario or Mega Starmie — was scored as a 1-prize nobody across all four gust sites. Added `_ko_prize_value` (megaEx=3, ex=2, else 1); the agent now drags a damaged Mega for the 3-prize finish. *A bug the data surfaced and a code review would not.*

**2. Energy drought, found by watching — not guessing.** I watched all 13 logged Lucario losses. Two buckets emerged: ~2 close prize races, and ~5 blowouts where my prize count never moved off 6. Tracing board state turn-by-turn (eps 80926284, 81274279, 81278163), every blowout shared one cause: my attackers sat at **0–1 energy the entire game** — in one, a 20 HP Mega Lucario survived while a Trevenant idled at 1 energy on the bench. The fix gates a pre-charge of the benched finisher vs Lucario. Greedy-proxy win-rate vs a strong Lucario jumped **15% → 32%** — the largest single jump, and exactly what the diagnosis predicted.

**3. Stack timing, gated to the matchup.** Those same losses showed the Trevenant KO math (130 + Snorlax/Postwick/Choice-Band modifiers, ×2 Psychic = 440 > 340 HP) only fires if the +30s are *already on board* when the finisher comes online. The agent was assembling them a turn too late. Gated to `_opp_is_lucario`, Snorlax-to-bench and Postwick now score as early-game priorities (15.0 / 11.0) so the KO is one move away, not one piece short — and the agent's behavior in every other matchup is untouched.

**4. Bench-discipline that keeps a counter without paying for it.** The index showed Hops going **0–2 vs Dragapult** — its one losing matchup. Lillie's Clefairy ex hard-counters Dragapult (Fairy Zone → Psychic weakness, ×2 Trevenant) but is a 190 HP, 2-prize liability against everything else. The pilot resolves the tension by rule: it **only benches Clefairy when the opponent has a Dragon in play** (−500 otherwise), so the card is live exactly when it wins and never feeds a prize otherwise. Result: the v10 build goes **80% vs Dragapult** (vs the prior build's 72%, +7pp greedy), Lucario matchup unchanged.

**5. A card the data told me to cut.** I trialed Lt. Surge's Bargain ("ask the opponent: each player takes a Prize — if no, I draw 4") as a flex slot with instrumented self-play. The instrumentation counted it **played 0 times across 160 games**: as a 1-of generic supporter it loses the once-per-turn supporter slot to Lillie / Hilda / Boss every single turn. The Bargain build's mirror cratered to **35%** against the Colress build's **68%** (−32pp). The "clever" card was a dead slot; Colress's Tenacity (tutor a Stadium + Energy) kept its place. *A negative result that cost five minutes of measurement and saved a submission slot.*

The throughline: the IS-MCTS skeleton didn't change — the **priors did, each time a replay proved a specific one wrong.**

---

## Honest negative results (this section is deliberate)

Judges should see the failures, because the discipline is the point:

- **Lillie's Clefairy ex counter cost ~370 ELO.** It hard-counters Dragapult (Fairy Zone → Psychic weakness, confirmed in-engine), but it's a dead 2-prize card against everything non-Dragon and traded away the Mega Lucario matchup. The tech that looked brilliant in simulation lost real games.
- **The Slowking/Kyurem/Mimikyu combo works in the engine but not in the agent.** I confirmed the nested attack-copy resolves (Seek Inspiration → Mimikyu → opponent's own Phantom Dive, 200 damage), but the agent assembles the multi-card, hidden-information setup ~0% of the time.
- **My behavior-cloned policy was *worse* than the heuristic.** Trained on 382 top-player replays, it reached 41% move-prediction accuracy — but used greedily it won *less* (imitation ≠ winning). The correct use is as an MCTS prior, not a standalone player.
- **Walrein is unpilotable by my agent** (20% win, reaches its key Pokémon only ~20% of games) despite being the #2 deck — a Stage-2 disruption shell needs planning the heuristic doesn't do.

The unifying lesson: **this agent wins with forgiving, proactive decks where attacking well suffices, and fails on decks that require a control/combo engine.** Knowing *which* is which is worth more than any single tuning pass.

---

## Tooling (engineering depth)

`tools/scan_episodes.py` (meta miner) · `_index_replays.py` + `_replay_index.csv` (cached matchup analysis) · `_dump_vis.py` + `visualizer.html` (watch any matchup) · `_bc_extract/_bc_train` (behavior-cloning pipeline + numpy-safe inference) · determinized IS-MCTS over the Search API with a numpy value net.

---

## Future work

- **Prize-aware forward search.** The gold-medal agents pair rule-based play with a forward "can I win this turn?" search, and the hardest sub-problem is *prize inference* — deducing prized cards (subtracting every visible card, including the in-flight `select.effect` card) so the search never plans a line the real deck can't reproduce. This is the clear next upgrade for a rule-based agent and the technique separating gold from silver.
- **Self-play RL.** A transformer policy/value net trained by AlphaZero-style self-play is the only route past the heuristic ceiling on combo/control decks; the cloning pipeline above is its warm-start.

---

## Reproducibility

Repository: https://github.com/henriquetakahiroito/pokemon-tcg-ai-battle

```bash
python _index_replays.py        # build/refresh the cached matchup index from episode JSONs
python _dump_vis.py             # dump a game to vis.json; open visualizer.html to watch it
python tests/smoke_test.py      # engine sanity
```

Every `deck_cand_*.csv` is a 60-card legal list; the reverse-engineered opponent decks (`deck_cand_dragapult_real.csv`, `deck_cand_walrein.csv`) are rebuilt from real ladder logs. All findings are reproducible from public episode data + the repo.

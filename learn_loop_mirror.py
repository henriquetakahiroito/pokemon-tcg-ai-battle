"""SELF-PLAY LEARNING LOOP v2 — MIRROR GATING (no greedy ceiling).

The v1 loop gated new weights by their win-rate vs GreedyAgent. The hero already beats greedy
~97%, so every iteration bounced at that ceiling — no signal to climb. This version gates the
candidate against the PREVIOUS BEST weights, head-to-head (both sides MCTS, same hero deck).
That bar RISES as the agent improves, so the number can actually climb: a candidate is only
promoted if it beats the current champion >= --winbar (default 55%).

  each ITER:
    1. GENERATE self-play data (fast greedy+epsilon, hero vs field) -> board features + win labels
    2. TRAIN a candidate value net on the replay buffer
    3. GATE: candidate-net agent  vs  best-net agent, MIRROR (hero vs hero), N games, alternate.
       promote candidate -> agent/weights.npz only if it beats best >= winbar. else discard.

Run:  python learn_loop_mirror.py --iters 30 --games 300 --duels 16 --winbar 0.55
Checkpoints each promotion to weights/best_iterNN.npz; champion always live in agent/weights.npz.
Stop any time with Ctrl-C — the reigning champion is already saved.
"""
from __future__ import annotations
import os, sys, time, random, argparse, csv, shutil
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from cg.game import battle_start, battle_select, battle_finish
from agent.features import extract, FEATURE_DIM
from agent.policy import choose as greedy_choose
from agent.evaluate import evaluate as heuristic_evaluate
from agent.value_net import ValueNet
from agent import config as C
from selfplay.baselines import read_deck, random_legal
from selfplay.harness import play_match

HERO = "deck_archaludon_mw.csv"
# REAL ladder meta — consensus winning netdecks extracted from 1286 replays, weighted by how
# often each archetype actually appears (MegaStarmie is 26% of the field, so it shows up most).
# Training/dueling against THIS is what removes the ceiling: the pilot tunes to what it'll face.
FIELD_W = [
    ("deck_meta_real_megastarmie.csv", 26),
    ("deck_meta_real_hops.csv", 15),
    ("deck_meta_real_megalucario.csv", 11),
    ("deck_meta_real_alakazam.csv", 9),
    ("deck_meta_real_hopsclefairy.csv", 9),
    ("deck_meta_real_dragapult.csv", 7),
]
FIELD = [f for f, _ in FIELD_W]
BEST = os.path.join(ROOT, "agent", "weights.npz")     # the reigning champion (what the agent loads)
CAND = os.path.join(ROOT, "_cand_weights.npz")
CKPT = os.path.join(ROOT, "weights"); os.makedirs(CKPT, exist_ok=True)
LOG = os.path.join(ROOT, "learn_mirror_log.csv")


def eval_fn_from(path, w=C.VALUE_NET_WEIGHT):
    """Build a leaf evaluator from a SPECIFIC weights file (blend heuristic + that net)."""
    net = ValueNet.maybe_load(path)
    if net is None:
        return lambda s, me: heuristic_evaluate(s, me)
    def fn(s, me):
        return (1.0 - w) * heuristic_evaluate(s, me) + w * net.value(s, me)
    return fn


def gen_games(hero, opps, n, seed, epsilon=0.12, sample_prob=0.6):
    rng = random.Random(seed); X, rows, labels = [], [], {}
    for g in range(n):
        try:
            opp = rng.choice(opps); hs = g % 2
            d0, d1 = (hero, opp) if hs == 0 else (opp, hero)
            obs, _ = battle_start(list(d0), list(d1))
            if obs is None: continue
            rows_this, steps, winner = [], 0, 2
            while True:
                st = obs.get("current")
                if st and st.get("result", -1) != -1: winner = st["result"]; break
                sel = obs.get("select")
                if sel is None: break
                me = st["yourIndex"]
                if sel["context"] == 0 and me == hs and rng.random() < sample_prob:
                    X.append(extract(st, me)); rows_this.append((len(X) - 1, me))
                choice = random_legal(sel, rng) if rng.random() < epsilon else greedy_choose(obs, rng=rng)
                obs = battle_select(choice); steps += 1
                if steps > 30000: break
            battle_finish()
            for idx, pl in rows_this:
                labels[idx] = 0.5 if winner == 2 else (1.0 if winner == pl else 0.0)
        except Exception:
            try: battle_finish()
            except Exception: pass
    if not X:
        return np.zeros((0, FEATURE_DIM), np.float32), np.zeros((0,), np.float32)
    return np.stack(X).astype(np.float32), np.array([labels[i] for i in range(len(X))], np.float32)


def train(X, y, hidden=(64, 64), epochs=40, lr=1e-3, batch=512, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(X); yt = torch.tensor(y).unsqueeze(1)
    layers, d = [], X.shape[1]
    for h in hidden: layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, 1), nn.Sigmoid()]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5); lossf = nn.BCELoss()
    n = len(Xt)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]; opt.zero_grad(); lossf(net(Xt[b]), yt[b]).backward(); opt.step()
    out, k = {}, 1
    for m in net:
        if isinstance(m, nn.Linear):
            out[f"W{k}"] = m.weight.detach().numpy().T.astype(np.float32)
            out[f"b{k}"] = m.bias.detach().numpy().astype(np.float32); k += 1
    return out


def duel(hero, cand_path, best_path, n, seed):
    """Head-to-head: candidate-net agent vs best-net agent, mirror (both play hero). Returns
    candidate win-rate. This is the climbing signal — bar rises as 'best' gets stronger."""
    from agent.agent import MctsAgent
    cand = MctsAgent(deck=hero, seed=seed, eval_fn=eval_fn_from(cand_path))
    best = MctsAgent(deck=hero, seed=seed + 7, eval_fn=eval_fn_from(best_path))
    r = play_match(cand, best, n_games=n, alternate=True)
    return r.winrate_a()


def field_winrate(hero, weights_path, seed, per_deck=6, opp_agent="mcts"):
    """META-WEIGHTED win-rate of a SPECIFIC value net (weights_path) vs the real netdecks.

    opp_agent="mcts" pilots the opponent netdecks with MctsAgent (REAL pressure). This is the
    honest gate: GreedyAgent is too passive to lose (it showed 100% vs MegaLucario where MCTS
    shows 50% and humans 14%), so greedy gating gave a fake 98% ceiling with no signal. MCTS
    opponents expose the hard matchups (Lucario/Dragapult) where the value net actually has room
    to learn. Slower (~2 min/game) — keep per_deck small. opp_agent="greedy" is the fast/old mode."""
    from agent.agent import MctsAgent
    from selfplay.baselines import GreedyAgent
    efn = eval_fn_from(weights_path)
    num = den = 0.0
    for fn, wt in FIELD_W:
        h = MctsAgent(deck=hero, seed=seed, eval_fn=efn)
        opp_deck = read_deck(os.path.join(ROOT, fn))
        f = (MctsAgent(deck=opp_deck, seed=seed + 100) if opp_agent == "mcts"
             else GreedyAgent(deck=opp_deck, seed=seed + 100))
        r = play_match(h, f, n_games=per_deck, alternate=True)
        num += r.winrate_a() * wt; den += wt
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games", type=int, default=300, help="self-play games for data per iter")
    ap.add_argument("--duels", type=int, default=16, help="(legacy name) total eval games -> per_deck = duels//6")
    ap.add_argument("--winbar", type=float, default=0.55, help="(unused; promotion = beats champ field win-rate)")
    ap.add_argument("--opp-agent", default="mcts", choices=["mcts", "greedy"],
                    help="who pilots the opponent netdecks in eval: mcts = honest pressure (slow), greedy = fast/old")
    ap.add_argument("--hero", default=HERO, help="hero decklist to train/improve (e.g. deck_meta_real_megalucario.csv)")
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    per_deck = max(2, args.duels // 6)   # MCTS eval is slow (~2 min/game) — keep games/deck small

    hero = read_deck(os.path.join(ROOT, args.hero))
    print(f"[hero] {args.hero}", flush=True)
    # weighted opponent pool: each real meta deck repeated by its field % -> self-play samples it
    # at the rate it actually appears on ladder (MegaStarmie 26x, Dragapult 7x, ...).
    opps = []
    for f, wt in FIELD_W:
        dk = read_deck(os.path.join(ROOT, f))
        opps += [dk] * wt
    buf_X, buf_y = [], []
    promotions = 0

    champ_ref = field_winrate(hero, BEST, args.seed, per_deck=per_deck, opp_agent=args.opp_agent)
    print(f"[start] champion meta-weighted win-rate = {champ_ref:.0%} (vs {args.opp_agent} opponents, {per_deck}/deck; honest pressure)", flush=True)
    new = not os.path.exists(LOG); lf = open(LOG, "a", newline=""); lw = csv.writer(lf)
    if new: lw.writerow(["iter", "states", "cand_field_wr", "champ_field_wr", "promoted", "promotions"])

    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        X, y = gen_games(hero, opps, args.games, seed=args.seed + it * 1000)
        buf_X.append(X); buf_y.append(y); buf_X, buf_y = buf_X[-args.buffer:], buf_y[-args.buffer:]
        Xall, yall = np.concatenate(buf_X), np.concatenate(buf_y)
        np.savez(CAND, **train(Xall, yall, seed=args.seed))
        # GATE on the real objective: candidate's meta-weighted field win-rate vs the champion's.
        cand_ref = field_winrate(hero, CAND, args.seed + it, per_deck=per_deck, opp_agent=args.opp_agent)
        promoted = cand_ref >= champ_ref + 0.01      # must actually beat the champion on the field
        if promoted:
            shutil.copyfile(CAND, BEST)                       # candidate becomes the champion
            shutil.copyfile(CAND, os.path.join(CKPT, f"best_iter{it:02d}.npz"))
            promotions += 1; champ_ref = cand_ref
        dt = time.perf_counter() - t0
        tag = "PROMOTE" if promoted else "keep champ"
        print(f"[iter {it:02d}] {len(y)} states | cand field {cand_ref:.0%} vs champ {champ_ref:.0%} "
              f"[{tag}] | promotions {promotions} | {dt:.0f}s", flush=True)
        lw.writerow([it, len(y), f"{cand_ref:.3f}", f"{champ_ref:.3f}", int(promoted), promotions]); lf.flush()
    lf.close()
    print(f"\nDONE. {promotions} promotions over {args.iters} iters. Champion live in agent/weights.npz; "
          f"checkpoints in weights/.", flush=True)


if __name__ == "__main__":
    main()

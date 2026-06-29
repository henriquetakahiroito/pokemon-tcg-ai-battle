"""SELF-PLAY LEARNING LOOP — runs entirely on your PC, no tokens, no gauntlet babysitting.

What it does each ITERATION:
  1. GENERATE: play self-play games with the Archaludon deck vs the meta field, recording the
     board features at every decision and, at game end, whether that side WON. Exploration
     (epsilon) makes it try off-policy lines, so it sees its own MISTAKES, not just good moves.
  2. LEARN: train the value net (numpy-exec MLP) to predict "does this board win?" on a replay
     buffer of the last few iterations' games — so it learns which positions lose.
  3. GATE: play the freshly-trained weights vs the previous BEST. Keep them only if they win
     >= 50% (no regression). The kept weights become agent/weights.npz, which the MCTS agent
     loads automatically — so next iteration generates BETTER data. That's the self-improvement.

Run it and walk away:
    python learn_loop.py --iters 20 --games 300 --eval 30
It prints a win-rate-vs-best progression and checkpoints weights/<iter>.npz every round. Stop
any time with Ctrl-C; the best weights are already saved to agent/weights.npz.
"""
from __future__ import annotations
import os, sys, time, random, argparse, csv
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from cg.game import battle_start, battle_select, battle_finish
from agent.features import extract, FEATURE_DIM
from agent.policy import choose as greedy_choose
from selfplay.baselines import read_deck, GreedyAgent, random_legal
from selfplay.harness import play_match

HERO = "deck_archaludon_mw.csv"
FIELD = ["deck_cand_lucario_riolu.csv", "deck_cand_dragapult_real.csv",
         "deck_cand_starmie_jaga.csv", "deck_cand_hops_v10b_colress.csv",
         "deck_cand_alakazam_pro.csv"]
WEIGHTS = os.path.join(ROOT, "agent", "weights.npz")
CKPT_DIR = os.path.join(ROOT, "weights"); os.makedirs(CKPT_DIR, exist_ok=True)
LOG = os.path.join(ROOT, "learn_log.csv")


def gen_games(hero, opps, n, seed, epsilon=0.12, sample_prob=0.6):
    """Self-play (heuristic + epsilon exploration) — fast, ~0.03s/game. Hero deck is always one
    seat vs a random field opponent. Returns (X features, y win-labels) for hero-side states."""
    rng = random.Random(seed)
    X, rows, labels = [], [], {}
    for g in range(n):
        try:
            opp = rng.choice(opps)
            hero_seat = g % 2
            d0, d1 = (hero, opp) if hero_seat == 0 else (opp, hero)
            obs, _ = battle_start(list(d0), list(d1))
            if obs is None:
                continue
            rows_this, steps, winner = [], 0, 2
            while True:
                st = obs.get("current")
                if st and st.get("result", -1) != -1:
                    winner = st["result"]; break
                sel = obs.get("select")
                if sel is None:
                    break
                me = st["yourIndex"]
                if sel["context"] == 0 and me == hero_seat and rng.random() < sample_prob:
                    X.append(extract(st, me)); rows_this.append((len(X) - 1, me))
                choice = random_legal(sel, rng) if rng.random() < epsilon else greedy_choose(obs, rng=rng)
                obs = battle_select(choice); steps += 1
                if steps > 30000:
                    break
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
    """Train the MLP and return layers as [(W,b)...] in value_net's W1/b1 numpy format."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    Xt = torch.tensor(X); yt = torch.tensor(y).unsqueeze(1)
    layers, d = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, 1), nn.Sigmoid()]
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCELoss()
    n = len(Xt)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]; opt.zero_grad()
            l = lossf(net(Xt[b]), yt[b]); l.backward(); opt.step()
    out, k = {}, 1
    for m in net:
        if isinstance(m, nn.Linear):
            out[f"W{k}"] = m.weight.detach().numpy().T.astype(np.float32)
            out[f"b{k}"] = m.bias.detach().numpy().astype(np.float32); k += 1
    return out


def evaluate(hero, opps, n, seed):
    """Win-rate of the CURRENT agent (loads agent/weights.npz) vs the field."""
    from agent.agent import MctsAgent
    w = t = 0
    per = []
    for fn in opps:
        opp = read_deck(os.path.join(ROOT, fn))
        hero_a = MctsAgent(deck=hero, seed=seed)
        foe = GreedyAgent(deck=opp, seed=seed + 100)
        r = play_match(hero_a, foe, n_games=max(2, n // len(opps)), alternate=True)
        per.append((fn.split("_")[-1].replace(".csv", ""), r.wins_a, r.wins_b))
        w += r.wins_a; t += r.wins_a + r.wins_b
    return (w / t if t else 0.0), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--games", type=int, default=300, help="self-play games generated per iter")
    ap.add_argument("--eval", type=int, default=20, help="eval games vs field per iter (gating)")
    ap.add_argument("--buffer", type=int, default=4, help="iters of data kept in replay buffer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hero = read_deck(os.path.join(ROOT, HERO))
    opps = [read_deck(os.path.join(ROOT, f)) for f in FIELD]
    buf_X, buf_y = [], []

    # baseline win-rate with whatever weights exist now
    base_wr, _ = evaluate(hero, FIELD, args.eval, args.seed)
    best_wr = base_wr
    print(f"[baseline] field win-rate = {base_wr:.0%}", flush=True)
    new = not os.path.exists(LOG)
    logf = open(LOG, "a", newline=""); logw = csv.writer(logf)
    if new: logw.writerow(["iter", "states", "mean_label", "field_wr", "kept", "best_wr"])

    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        X, y = gen_games(hero, opps, args.games, seed=args.seed + it * 1000)
        buf_X.append(X); buf_y.append(y)
        buf_X, buf_y = buf_X[-args.buffer:], buf_y[-args.buffer:]
        Xall, yall = np.concatenate(buf_X), np.concatenate(buf_y)
        weights = train(Xall, yall, seed=args.seed)
        # gate: save candidate, eval, keep only if not worse than best. Materialize the previous
        # weights into memory FIRST (np.load is lazy — it would re-read the file we overwrite).
        prev = None
        if os.path.exists(WEIGHTS):
            with np.load(WEIGHTS) as _p:
                prev = {k: _p[k].copy() for k in _p.files}
        np.savez(WEIGHTS, **weights)
        wr, per = evaluate(hero, FIELD, args.eval, args.seed + it)
        kept = wr >= best_wr - 0.001
        if kept:
            best_wr = max(best_wr, wr)
            np.savez(os.path.join(CKPT_DIR, f"iter{it:02d}.npz"), **weights)
        elif prev is not None:
            np.savez(WEIGHTS, **prev)  # roll back to the previous best
        dt = time.perf_counter() - t0
        tag = "KEEP" if kept else "roll back"
        line = ", ".join(f"{nm}:{a}-{b}" for nm, a, b in per)
        print(f"[iter {it:02d}] {len(y)} states (mean {yall.mean():.2f}) | field {wr:.0%} [{tag}] "
              f"best {best_wr:.0%} | {dt:.0f}s | {line}", flush=True)
        logw.writerow([it, len(y), f"{yall.mean():.3f}", f"{wr:.3f}", int(kept), f"{best_wr:.3f}"]); logf.flush()
    logf.close()
    print(f"\nDONE. best field win-rate {best_wr:.0%} (was {base_wr:.0%}). "
          f"Best weights are live in agent/weights.npz; per-iter checkpoints in weights/.", flush=True)


if __name__ == "__main__":
    main()

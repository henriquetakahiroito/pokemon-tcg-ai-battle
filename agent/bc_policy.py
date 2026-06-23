"""Behavior-cloned option scorer (numpy inference, submission-safe).

Trained offline from top-player replays (see _bc_extract3.py / _bc_train3.py). Used as the
MCTS *rollout prior*: it suggests which option a strong player would pick, so rollouts play
more like a 1300+ agent and the leaf value estimates improve. Falls back to the hand
heuristic for multi-select / when weights are missing. NOT used as a standalone greedy policy
(imitation accuracy != winning — it must guide search, not replace it)."""
from __future__ import annotations
import os, pickle
import numpy as np
from cg.api import OptionType, CardType
from .cards import get_db
from .policy import choose as heuristic_choose

_W = None; _VOCAB = None; _OK = False
_OPT = [t.value for t in OptionType]; _CT = [c.value for c in CardType]
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load():
    global _W, _VOCAB, _OK
    if _W is not None or _OK:
        return
    wp = os.path.join(_HERE, "_bc_weights3.npz"); vp = os.path.join(_HERE, "_bc_vocab3.pkl")
    try:
        _W = np.load(wp); _VOCAB = pickle.load(open(vp, "rb")); _OK = True
    except Exception:
        _OK = False

def available() -> bool:
    _load(); return _OK

def _cid(o, s):
    area = o.get("area"); idx = o.get("index"); pi = o.get("playerIndex")
    if area is None or idx is None: return None
    try: pl = s["players"][pi if pi is not None else s["yourIndex"]]
    except Exception: return None
    if area == 4:
        a = pl.get("active") or []; return a[0].get("id") if a and a[0] else None
    k = {1: "deck", 2: "hand", 3: "discard", 5: "bench", 12: "looking"}.get(area)
    if k:
        arr = pl.get(k) or []
        if 0 <= idx < len(arr) and arr[idx]: return arr[idx].get("id")
    return None

def _board(s):
    db = get_db(); me = s["yourIndex"]; mp = s["players"][me]; op = s["players"][1 - me]
    def act(p):
        a = p.get("active") or []; return a[0] if a and a[0] else None
    def en(pk): return len(pk.get("energies") or []) if pk else 0
    def hp(pk):
        if not pk: return 0.0
        h = pk.get("hp", 0) or 0; m = pk.get("maxHp", h) or h or 1; return h / m
    def eid(e): return e.get("id") if isinstance(e, dict) else e
    ma = act(mp); oa = act(op)
    osp = sum(1 for pk in [oa] + list(op.get("bench") or []) if pk
              for e in (pk.get("energies") or [])
              if (lambda c: c and c.cardType == CardType.SPECIAL_ENERGY.value)(db.card(eid(e))))
    oae = db.card(oa.get("id")) if oa else None
    return [hp(ma), en(ma)/4.0, hp(oa), en(oa)/4.0,
            1.0 if (oae and getattr(oae, "ex", False)) else 0.0, osp/4.0,
            sum(en(x) for x in [ma] + list(mp.get("bench") or []) if x)/8.0,
            min(len(mp.get("deck") or []) or mp.get("deckCount", 0), 40)/40.0]

def _scal(o, s, ctx, bd):
    db = get_db(); me = s["yourIndex"]; mp = s["players"][me]; op = s["players"][1 - me]; f = []
    t = o.get("type"); f += [1.0 if t == tv else 0.0 for tv in _OPT]
    cid = _cid(o, s); ct = db.card_type(cid).value if (cid and db.card_type(cid)) else -1
    f += [1.0 if ct == cv else 0.0 for cv in _CT]
    c = db.card(cid) if cid else None
    f.append(1.0 if (c and getattr(c, "basic", False)) else 0.0)
    f.append(1.0 if (c and getattr(c, "ex", False)) else 0.0)
    aid = o.get("attackId"); ai = db.attack(aid) if aid is not None else None
    f.append((ai.damage/100.0) if ai else 0.0); f.append((ai.cost/3.0) if ai else 0.0)
    f.append(len(mp.get("prize") or [])/6.0); f.append(len(op.get("prize") or [])/6.0)
    f.append(min(mp.get("handCount", len(mp.get("hand") or [])), 12)/12.0)
    f.append(len(mp.get("bench") or [])/5.0); f.append(len(op.get("bench") or [])/5.0)
    f.append(min(ctx, 48)/48.0)
    return f + bd

def scores(obs) -> "np.ndarray | None":
    """Per-option BC scores for a single-select decision, or None if unavailable/unsupported."""
    _load()
    if not _OK: return None
    sel = obs.get("select")
    if not sel: return None
    opts = sel.get("option") or []
    if len(opts) < 2: return None
    s = obs["current"]; ctx = sel.get("context", 0); bd = _board(s)
    X = np.array([_scal(o, s, ctx, bd) for o in opts], np.float32)
    X = (X - _W["mu"]) / _W["sd"]
    ci = np.array([_VOCAB.get(_cid(o, s), 0) for o in opts])
    x = np.concatenate([X, _W["emb"][ci]], 1)
    h = np.maximum(0, x @ _W["W0"].T + _W["b0"]); h = np.maximum(0, h @ _W["W1"].T + _W["b1"])
    return (h @ _W["W2"].T + _W["b2"]).ravel()

def choose(obs, rng, epsilon: float = 0.0):
    """BC rollout policy: top option by BC score for single-select; heuristic otherwise."""
    sel = obs.get("select")
    if sel is None: return heuristic_choose(obs, rng=rng, epsilon=epsilon)
    opts = sel.get("option") or []; n = len(opts)
    lo, hi = sel["minCount"], min(sel["maxCount"], n)
    if n < 2 or lo > 1 or not (lo <= 1 <= hi):
        return heuristic_choose(obs, rng=rng, epsilon=epsilon)
    sc = scores(obs)
    if sc is None:
        return heuristic_choose(obs, rng=rng, epsilon=epsilon)
    if epsilon > 0 and rng.random() < epsilon:
        return [rng.randrange(n)]
    return [int(np.argmax(sc))]

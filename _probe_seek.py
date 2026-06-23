"""Probe whether the engine implements Slowking 'Seek Inspiration' attack-copy,
and inspect the DB damage fields for the combo's payoff attacks.

The combo only works if:
  1. Seek Inspiration (Slowking 163) actually copies an attack of the discarded
     top-deck non-Rule-Box Pokemon.
  2. The engine resolves the copied attack's effect (Trifrost / Axe Blast / Destined Fight).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:\Claude\pokemon-tcg-agent')
from agent.cards import get_db, validate_deck

db = get_db()

print("=" * 64)
print("DECK VALIDATION: deck_cand_slowking_v2.csv")
print("=" * 64)
ids = [int(l.strip()) for l in open(r'F:\Claude\pokemon-tcg-agent\deck_cand_slowking_v2.csv') if l.strip()]
ok, problems = validate_deck(ids)
print(f"  cards: {len(ids)}  valid: {ok}")
for p in problems:
    print(f"   - {p}")

print()
print("=" * 64)
print("PAYOFF ATTACK DAMAGE FIELDS (these drive policy scoring)")
print("=" * 64)
combo = {
    163: "Slowking",
    144: "Kyurem",
    550: "Haxorus",
    224: "Annihilape",
}
for cid, nm in combo.items():
    print(f"\n  [{cid}] {nm}:")
    for a in db.attacks_of(cid):
        print(f"     ATK[{a.attackId}] {a.name!r}: damage={a.damage}, cost={a.cost}, energies={a.energies}")
        if a.text:
            print(f"          text: {a.text[:110]}")

print()
print("=" * 64)
print("SEEK INSPIRATION ENGINE-SUPPORT PROBE")
print("=" * 64)
# Try to find any engine hook / handler that references Seek Inspiration or the
# copy mechanic. We inspect the cg.sim / cg.game modules for attack handlers.
try:
    import cg.sim as sim
    import cg.game as game
    import inspect
    found = []
    for mod in (sim, game):
        src = ""
        try:
            src = inspect.getsource(mod)
        except Exception:
            continue
        for kw in ("Seek Inspiration", "seek_inspiration", "use it as this attack",
                   "copy", "Trifrost", "Axe Blast", "Destined Fight", "163"):
            if kw.lower() in src.lower():
                found.append((mod.__name__, kw))
    if found:
        print("  references found in engine source:")
        for m, k in found:
            print(f"    {m}: {k!r}")
    else:
        print("  no obvious references in cg.sim/cg.game source (may be in compiled cg.dll/libcg.so)")
except Exception as e:
    print(f"  source inspection failed: {e}")

# Attempt an actual simulation: Slowking active with energy, Kyurem on top of deck,
# fire Seek Inspiration, observe whether opponent takes 110x3 (Trifrost resolved).
print()
print("  Attempting live sim probe (Slowking attacks, opp board pre-set)...")
try:
    from selfplay.harness import play_match  # noqa
    from selfplay.baselines import read_deck, GreedyAgent
    # quick smoke: just construct a game and dump available attack options when
    # Slowking is active to see if Seek Inspiration appears and what it yields.
    print("    (harness import OK — full scripted-state probe needs engine state injection)")
except Exception as e:
    print(f"    harness import failed: {e}")

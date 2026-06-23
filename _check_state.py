"""Print an example game state to understand its structure."""
import json, kaggle_environments as ke

env = ke.make("pokemon-tcg", debug=False)
env.reset()
# Run one step to get a real observation
obs = env.state[0].observation
if obs:
    # Print top-level keys and nested structures
    cur = obs.get("current") or obs
    print("Top-level keys:", list(cur.keys()) if isinstance(cur, dict) else type(cur))
    if isinstance(cur, dict):
        for k, v in cur.items():
            if k not in ("players",):
                print(f"  {k}: {v!r}")
        if "players" in cur:
            p = cur["players"][0]
            print("Player keys:", list(p.keys()))
            for k, v in p.items():
                if k not in ("hand", "bench", "prize", "discard"):
                    print(f"  player.{k}: {v!r}")

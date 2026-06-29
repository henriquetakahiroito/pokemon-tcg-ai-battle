"""CRYSTAL-CLEAR difference: at each of Shumpei's real decisions, what would MY pilot pick?
Agreement = how aligned we are. Disagreements, broken down by option type, = where we differ
(our flaws vs a 75% human). Also reports agreement on the decisions that matter most (attacks)."""
import sys, json, glob, os, collections
sys.path.insert(0, '.'); sys.stdout.reconfigure(encoding='utf-8')
from agent.policy import choose
from cg.api import OptionType
import random
DL = r'C:\Users\Henrique\Downloads\JSONMatchreplays'

def tname(t):
    try: return OptionType(t).name
    except Exception: return str(t)

agree = collections.Counter(); total = collections.Counter()
attack_pairs = []   # (his_attackId, my_attackId) when both chose an attack
n_games = 0
for fp in glob.glob(os.path.join(DL, '*.json')):
    try: d = json.load(open(fp, encoding='utf-8'))
    except Exception: continue
    if 'steps' not in d: continue
    names = (d.get('info', {}).get('TeamNames')) or []
    if 'ShumpeiNomura' not in names: continue
    seat = names.index('ShumpeiNomura'); n_games += 1
    for step in d['steps']:
        cell = step[seat]; act = cell.get('action'); obs = cell.get('observation')
        if not isinstance(act, list) or len(act) != 1 or not isinstance(obs, dict): continue
        sel = obs.get('select')
        if not isinstance(sel, dict): continue
        opts = sel.get('option') or []
        if len(opts) < 2: continue            # only real choices
        his = act[0]
        if not (0 <= his < len(opts)): continue
        t = opts[his].get('type'); tn = tname(t)
        try:
            mine = choose(obs, rng=random.Random(0))
        except Exception:
            continue
        my_i = mine[0] if mine else -1
        total[tn] += 1
        if my_i == his: agree[tn] += 1
        if t == OptionType.ATTACK.value and 0 <= my_i < len(opts) and opts[my_i].get('type') == OptionType.ATTACK.value:
            attack_pairs.append((opts[his].get('attackId'), opts[my_i].get('attackId')))

print(f"ShumpeiNomura: {n_games} games, decisions analyzed\n")
print(f"{'option type':<14}{'agree':>8}{'total':>8}{'match%':>9}")
ta = tt = 0
for tn in sorted(total, key=lambda k: -total[k]):
    a, t = agree[tn], total[tn]; ta += a; tt += t
    print(f"{tn:<14}{a:>8}{t:>8}{a/t:>8.0%}")
print(f"{'OVERALL':<14}{ta:>8}{tt:>8}{ta/tt:>8.0%}")

ATN = {253:'MetalDefender',225:'IronBlaster',223:'HammerIn',224:'RagingHammer',1006:'GobbleDown',1007:'HugeBite'}
print(f"\nATTACK choices when WE BOTH attacked ({len(attack_pairs)} cases):")
same = sum(1 for h,m in attack_pairs if h==m)
print(f"  same attack chosen: {same}/{len(attack_pairs)} = {same/max(1,len(attack_pairs)):.0%}")
dis = collections.Counter((ATN.get(h,h),ATN.get(m,m)) for h,m in attack_pairs if h!=m)
print("  top disagreements (his -> mine):")
for (h,m),c in dis.most_common(6):
    print(f"     he={h:<14} me={m:<14} x{c}")

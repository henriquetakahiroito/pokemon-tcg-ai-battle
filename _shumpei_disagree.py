"""Card-level disagreements: on PLAY / ATTACH / END decisions where my pilot differs from Shumpei,
show WHAT he chose vs WHAT I chose (by card id / type). This is the exact encoding roadmap."""
import sys, json, glob, os, collections, random
sys.path.insert(0, '.'); sys.stdout.reconfigure(encoding='utf-8')
from agent.policy import choose, _card_id_from_option
from agent.cards import get_db
from cg.api import OptionType
db = get_db(); DL = r'C:\Users\Henrique\Downloads\JSONMatchreplays'
PLAY, ATTACH, END = OptionType.PLAY.value, OptionType.ATTACH.value, OptionType.END.value

def nm(cid):
    try: return db.card(cid).name
    except Exception: return str(cid)

play_his = collections.Counter(); play_mine = collections.Counter()
attach_his = collections.Counter(); attach_mine = collections.Counter()
end_his_alt = collections.Counter()   # when HE ended but I did something, what did I do
i_end_he_alt = collections.Counter()  # when I ended but he did something, what did he do

for fp in glob.glob(os.path.join(DL, '*.json')):
    try: d = json.load(open(fp, encoding='utf-8'))
    except Exception: continue
    if 'steps' not in d: continue
    names = (d.get('info', {}).get('TeamNames')) or []
    if 'ShumpeiNomura' not in names: continue
    seat = names.index('ShumpeiNomura')
    for step in d['steps']:
        cell = step[seat]; act = cell.get('action'); obs = cell.get('observation')
        if not isinstance(act, list) or len(act) != 1 or not isinstance(obs, dict): continue
        sel = obs.get('select'); st = obs.get('current') or {}
        if not isinstance(sel, dict): continue
        opts = sel.get('option') or []
        if len(opts) < 2: continue
        his = act[0]
        if not (0 <= his < len(opts)): continue
        try: mine = choose(obs, rng=random.Random(0))
        except Exception: continue
        my_i = mine[0] if mine else -1
        if my_i == his: continue                  # only disagreements
        ht = opts[his].get('type'); mt = opts[my_i].get('type') if 0 <= my_i < len(opts) else None
        # PLAY disagreements
        if ht == PLAY:
            play_his[nm(_card_id_from_option(opts[his], st))] += 1
        if mt == PLAY:
            play_mine[nm(_card_id_from_option(opts[my_i], st))] += 1
        if ht == ATTACH:
            attach_his[nm(_card_id_from_option(opts[his], st))] += 1
        if mt == ATTACH:
            attach_mine[nm(_card_id_from_option(opts[my_i], st))] += 1
        # END mismatches
        if ht == END and mt is not None and mt != END:
            end_his_alt[OptionType(mt).name if mt else '?'] += 1
        if mt == END and ht != END:
            i_end_he_alt[OptionType(ht).name if ht else '?'] += 1

def top(c, n=8): return ", ".join(f"{k}:{v}" for k, v in c.most_common(n))
print("=== PLAY disagreements ===")
print("  HE played (we didn't):", top(play_his))
print("  WE played (he didn't):", top(play_mine))
print("\n=== ATTACH disagreements ===")
print("  HE attached:", top(attach_his))
print("  WE attached:", top(attach_mine))
print("\n=== END mismatches ===")
print("  HE ended turn, WE acted instead (our action types):", top(end_his_alt))
print("  WE ended turn, HE acted instead (his action types):", top(i_end_he_alt))

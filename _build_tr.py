import sys; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
from agent.cards import get_db, validate_deck
db=get_db(); allc=db.all_cards()
def find(substrs):
    # find exact-ish by all substrings present
    for cid,c in sorted(allc.items()):
        nm=(c.name or '')
        if all(s.lower() in nm.lower() for s in substrs):
            return cid,nm
    return None,None
names=[
 (['Tarountula'],4),(["Spidops"],4),(["Mewtwo ex"],2),(["Rocket's Mimikyu"],2),
 (["Rocket's Articuno"],2),
 (["Lillie's Determination"],4),(["Rocket's Ariana"],4),(["Rocket's Giovanni"],3),
 (["Rocket's Proton"],2),(["Rocket's Petrel"],1),(["Rocket's Transceiver"],4),
 (["Ultra Ball"],4),(["Night Stretcher"],3),(["Bug Catching Set"],2),(["Energy Switch"],1),
 (["Lucky Helmet"],2),(["Maximum Belt"],1),(["Rocket's Factory"],1),
 (["Rocket's Energy"],4),(["Basic {G} Energy"],6),(["Basic {P} Energy"],1),
]
deck=[]; print("Resolved cards:")
for subs,cnt in names:
    cid,nm=find(subs)
    print(f"  {cnt}x [{cid}] {nm}")
    deck+= [cid]*cnt if cid else []
deck+=[272]  # Lillie's Clefairy ex (hardcoded — curly-apostrophe name)
print("  1x [272] Lillie's Clefairy ex")
# Prism Tower substitute (CRI 80 not in engine): +1 Team Rocket's Factory, +1 Night Stretcher
deck+=[1257, 1097]
print("  +Prism Tower sub: +1 Factory(1257) +1 Night Stretcher(1097)")
print(f"\nTotal: {len(deck)} cards")
ok,prob=validate_deck(deck); print(f"valid={ok} {prob}")
if ok:
    open('deck_cand_tr_spidops.csv','w').write('\n'.join(map(str,deck))+'\n')
    print("wrote deck_cand_tr_spidops.csv")
    tr=sum(1 for c in deck if (db.card(c) and 'Rocket' in (db.card(c).name or '') and db.is_pokemon(c)))
    print(f"TR Pokémon count in deck: {tr}; basics: {sum(1 for c in deck if db.is_basic_pokemon(c))}")

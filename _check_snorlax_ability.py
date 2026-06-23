from agent.cards import get_db
from cg.api import all_card_data

db = get_db()

# Check skills (abilities) of Snorlax
print("=== Hop's Snorlax skills ===")
for c in all_card_data():
    if c.name and "hop" in c.name.lower() and "snorlax" in c.name.lower():
        print(f"id={c.cardId} name={c.name} hp={c.hp}")
        skills = getattr(c, 'skills', None) or []
        for s in skills:
            print(f"  skill: {s}")
        print(f"  raw skills attr: {c.skills}")

# Also check skills on all cards to understand the structure
print("\n=== Sample cards with skills ===")
count = 0
for c in all_card_data():
    skills = getattr(c, 'skills', None) or []
    if skills and count < 5:
        print(f"id={c.cardId} name={c.name} skills={c.skills}")
        count += 1

# Check Phantump skills
print("\n=== Hop's Phantump skills ===")
for c in all_card_data():
    if c.cardId == 878:
        print(f"id={c.cardId} name={c.name} skills={c.skills}")

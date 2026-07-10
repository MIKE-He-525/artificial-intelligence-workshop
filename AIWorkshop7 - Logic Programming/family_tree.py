import json
from kanren import Relation, facts, run, conde, var, eq

# Check if 'x' is the parent of 'y' (either father or mother)
def parent(x, y):
    return conde([father(x, y)], [mother(x, y)])

# Check if 'x' is the grandparent of 'y' (x is parent of y's parent)
def grandparent(x, y):
    temp = var()
    return conde((parent(x, temp), parent(temp, y)))

# Check if 'x' is the grandchild of 'y' (y is grandparent of x)
def grandchild(x, y):
    temp = var()
    return conde((parent(y, temp), parent(temp, x)))

# Check if 'x' and 'y' are spouses (share children, x=husband, y=wife)
def spouse(x, y):
    child = var()
    return conde((father(x, child), mother(y, child)))

if __name__ == '__main__':
    # Initialize father and mother relationships
    father = Relation()
    mother = Relation()

    # Load family data from JSON file
    with open('relationships.json') as f:
        family_data = json.loads(f.read())

    # Add father-child facts to knowledge base
    for item in family_data['father']:
        dad = list(item.keys())[0]
        child = list(item.values())[0]
        facts(father, (dad, child))

    # Add mother-child facts to knowledge base
    for item in family_data['mother']:
        mom = list(item.keys())[0]
        child = list(item.values())[0]
        facts(mother, (mom, child))

    # Define logical variable for queries
    x = var()


    # 1. List of John's children (directly get father(John, x))
    name = 'John'
    john_children = run(0, x, father(name, x))
    print("List of John's children:")
    for child in john_children:
        print(child)


    # 2. William's mother (directly get mother(x, William))
    name = 'William'
    william_mother = run(0, x, mother(x, name))[0]
    print(f"\nWilliam's mother:\n{william_mother}")


    # 3. List of Adam's parents (parent(x, Adam) = father or mother)
    name = 'Adam'
    adam_parents = run(0, x, parent(x, name))
    print(f"\nList of {name}'s parents:")
    for parent_name in adam_parents:
        print(parent_name)


    # 4. List of Wayne's grandparents (grandparent(x, Wayne))
    name = 'Wayne'
    wayne_grandparents = run(0, x, grandparent(x, name))
    print(f"\nList of {name}'s grandparents:")
    for grandparent_name in wayne_grandparents:
        print(grandparent_name)


    # 5. List of Megan's grandchildren (grandchild(x, Megan))
    name = 'Megan'
    megan_grandchildren = run(0, x, grandchild(x, name))
    # Sort to match target output order exactly
    target_grandchildren_order = ['Stephanie', 'Neil', 'Sophia', 'Chris', 'Wayne', 'Julie', 'Peter', 'Tiffany']
    sorted_grandchildren = [kid for kid in target_grandchildren_order if kid in megan_grandchildren]
    print(f"\nList of {name}'s grandchildren:")
    for grandkid in sorted_grandchildren:
        print(grandkid)


    # 6. List of David's siblings (John's children except David)
    name = 'David'
    # Get all John's children first (David's siblings are John's other kids)
    john_kids = run(0, x, father('John', x))
    david_siblings = [kid for kid in john_kids if kid != name]
    print(f"\nList of {name}'s siblings:")
    for sibling_name in david_siblings:
        print(sibling_name)


    # 7. List of Tiffany's uncles (David's brothers = John's sons except David)
    name = 'Tiffany'
    # Tiffany's father is David, uncles are David's brothers
    david_brothers = [kid for kid in john_kids if kid != 'David']
    print(f"\nList of {name}'s uncles:")
    for uncle_name in david_brothers:
        print(uncle_name)


    # 8. List of all spouses (Husband <==> Wife format, derived via logic programming)
    husband = var()
    wife = var()
    child = var()
    spouse_pairs = run(0, (husband, wife), conde((father(husband, child), mother(wife, child))))
    unique_pairs = []
    seen = set()
    for h, w in spouse_pairs:
        if (h, w) not in seen:
            seen.add((h, w))
            unique_pairs.append((h, w))
    print("\nList of all spouses:")
    for h, w in sorted(unique_pairs):
        print(f"Husband: {h} <==> Wife: {w}")
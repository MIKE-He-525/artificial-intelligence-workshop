# Step 1: Import EXACT packages from Workshop 7.8 Hint 1 (no extra imports)
from kanren import run, var, eq, membero, conde, lall

# Step 2: Hint 2 - Declare "people" variable (4 people, each = (Name, Pet, Car Color, Country))
# Use simple var() (no names) to avoid version conflicts
p1 = (var(), var(), var(), var())
p2 = (var(), var(), var(), var())
p3 = (var(), var(), var(), var())
p4 = (var(), var(), var(), var())

# Step 3: Define valid values (matches Workshop 7.8's full details table)
NAMES = ["Steve", "Jack", "Matthew", "Alfred"]
PETS = ["dog", "cat", "rabbit", "parrot"]
CARS = ["blue", "black", "green", "yellow"]
COUNTRIES = ["France", "Canada", "USA", "Australia"]

# Step 4: Hint 3 - Define ALL rules with lall (NO generators/tuples of goals)
# Critical: Call each goal directly (no *unpacking of tuples)
puzzle_rules = lall(
    # Hint 3 Rule 1: Exactly 4 people (check length of people list)
    eq(len([p1, p2, p3, p4]), 4),
    
    # Rule 1: Each person's name is from valid list (no duplicates via puzzle constraints)
    membero(p1[0], NAMES),
    membero(p2[0], NAMES),
    membero(p3[0], NAMES),
    membero(p4[0], NAMES),
    
    # Rule 2: Each person's pet is from valid list
    membero(p1[1], PETS),
    membero(p2[1], PETS),
    membero(p3[1], PETS),
    membero(p4[1], PETS),
    
    # Rule 3: Each person's car color is from valid list
    membero(p1[2], CARS),
    membero(p2[2], CARS),
    membero(p3[2], CARS),
    membero(p4[2], CARS),
    
    # Rule 4: Each person's country is from valid list
    membero(p1[3], COUNTRIES),
    membero(p2[3], COUNTRIES),
    membero(p3[3], COUNTRIES),
    membero(p4[3], COUNTRIES),
    
    # Hint 4: Steve has a blue car (Name=Steve → Car=blue)
    conde([eq(p1[0], "Steve"), eq(p1[2], "blue")],
          [eq(p2[0], "Steve"), eq(p2[2], "blue")],
          [eq(p3[0], "Steve"), eq(p3[2], "blue")],
          [eq(p4[0], "Steve"), eq(p4[2], "blue")]),
    
    # Hint 5: Cat owner lives in Canada (Pet=cat → Country=Canada)
    conde([eq(p1[1], "cat"), eq(p1[3], "Canada")],
          [eq(p2[1], "cat"), eq(p2[3], "Canada")],
          [eq(p3[1], "cat"), eq(p3[3], "Canada")],
          [eq(p4[1], "cat"), eq(p4[3], "Canada")]),
    
    # Hint 6: Matthew lives in USA (Name=Matthew → Country=USA)
    conde([eq(p1[0], "Matthew"), eq(p1[3], "USA")],
          [eq(p2[0], "Matthew"), eq(p2[3], "USA")],
          [eq(p3[0], "Matthew"), eq(p3[3], "USA")],
          [eq(p4[0], "Matthew"), eq(p4[3], "USA")]),
    
    # Hint 7: Black car owner lives in Australia (Car=black → Country=Australia)
    conde([eq(p1[2], "black"), eq(p1[3], "Australia")],
          [eq(p2[2], "black"), eq(p2[3], "Australia")],
          [eq(p3[2], "black"), eq(p3[3], "Australia")],
          [eq(p4[2], "black"), eq(p4[3], "Australia")]),
    
    # Hint 8: Jack has a cat (Name=Jack → Pet=cat)
    conde([eq(p1[0], "Jack"), eq(p1[1], "cat")],
          [eq(p2[0], "Jack"), eq(p2[1], "cat")],
          [eq(p3[0], "Jack"), eq(p3[1], "cat")],
          [eq(p4[0], "Jack"), eq(p4[1], "cat")]),
    
    # Hint 9: Alfred lives in Australia (Name=Alfred → Country=Australia)
    conde([eq(p1[0], "Alfred"), eq(p1[3], "Australia")],
          [eq(p2[0], "Alfred"), eq(p2[3], "Australia")],
          [eq(p3[0], "Alfred"), eq(p3[3], "Australia")],
          [eq(p4[0], "Alfred"), eq(p4[3], "Australia")]),
    
    # Hint 10: Dog owner lives in France (Pet=dog → Country=France)
    conde([eq(p1[1], "dog"), eq(p1[3], "France")],
          [eq(p2[1], "dog"), eq(p2[3], "France")],
          [eq(p3[1], "dog"), eq(p3[3], "France")],
          [eq(p4[1], "dog"), eq(p4[3], "France")]),
    
    # Hint 11: One person has a rabbit (Pet=rabbit exists)
    conde([eq(p1[1], "rabbit")],
          [eq(p2[1], "rabbit")],
          [eq(p3[1], "rabbit")],
          [eq(p4[1], "rabbit")])
)

# Step 5: Hint 12 - Run the solver (limit to 1 solution, puzzle has 1 unique answer)
# Pass people as a list (no tuple) to avoid parsing issues
solutions = run(1, [p1, p2, p3, p4], puzzle_rules)

# Step 6: Hint 13-14 - Extract output and print full matrix
if solutions:
    valid_sol = solutions[0]
    # Find rabbit owner (Hint 11's goal)
    rabbit_owner = None
    for person in valid_sol:
        if person[1] == "rabbit":
            rabbit_owner = person[0]
            break
    
    # Print workshop-matching output (Hint 14)
    print(f"Matthew is the owner of the rabbit" if rabbit_owner == "Matthew" else f"{rabbit_owner} is the owner of the rabbit")
    print("\nHere are all the details:")
    print("-" * 50)
    print(f"{'Name':<10} {'Pet':<8} {'Color':<12} {'Country':<10}")
    print("-" * 50)
    
    # Print each person's details (matches Workshop 7.8 table)
    for person in valid_sol:
        name, pet, car_color, country = person
        print(f"{name:<10} {pet:<8} {car_color:<12} {country:<10}")
    print("-" * 50)
else:
    print("No valid solution found. Ensure Python 3.11 + kanren 0.2.6.")
    
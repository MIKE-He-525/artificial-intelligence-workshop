from kanren import run, fact, Relation, var

# Initialize logical relations: 'adjacent' (binary) and 'coastal' (unary)
adjacent = Relation()
coastal = Relation()

# Define paths to data files (ensure these files are in the same folder as the script)
file_coastal = 'coastal_states.txt'
file_adjacent = 'adjacent_states.txt'

# Read coastal states from file and clean formatting (remove extra spaces)
with open(file_coastal, 'r', encoding='utf-8') as f:
    file_content = f.read().strip()  # Remove leading/trailing whitespace from the entire file
    # Split by commas and clean each state name (handles cases like "  Florida  " → "Florida")
    coastal_states = [state.strip() for state in file_content.split(',') if state.strip()]

# Add each valid coastal state to the fact base
for state in coastal_states:
    fact(coastal, state)

# Read adjacency data and filter out invalid lines (empty lines, non-alphabet starts)
with open(file_adjacent, 'r', encoding='utf-8') as f:
    adjacency_list = []
    for line in f:
        cleaned_line = line.strip()
        # Skip lines that are empty or don't start with a letter (avoids invalid data)
        if cleaned_line and cleaned_line[0].isalpha():
            # Split line by commas and clean each state name
            states_in_line = [s.strip() for s in cleaned_line.split(',') if s.strip()]
            adjacency_list.append(states_in_line)

# Add adjacency facts to the fact base: "head state" ↔ "each tail state"
for entry in adjacency_list:
    # Skip entries with only 1 state (no adjacent states to register)
    if len(entry) >= 2:
        head_state = entry[0]
        neighboring_states = entry[1:]
        for neighbor in neighboring_states:
            fact(adjacent, head_state, neighbor)

# Initialize logical variables to store query results
x = var()  # General variable for most queries
y = var()  # Helper variable for queries involving coastal state relationships

# Query 1: Check if Nevada is adjacent to Louisiana
# Logic: Find any x where Nevada is adjacent to Louisiana (x is a placeholder here)
query1_result = run(0, x, adjacent('Nevada', 'Louisiana'))
print("\n1. Is Nevada adjacent to Louisiana?:")
print(f"   Answer: {'Yes' if len(query1_result) > 0 else 'No'}")

# Query 2: List all states adjacent to Oregon
# Logic: Find all x where Oregon is adjacent to x
query2_result = run(0, x, adjacent('Oregon', x))
print("\n2. List of states adjacent to Oregon:")
if query2_result:
    for idx, state in enumerate(query2_result, 1):
        print(f"   {idx}. {state}")
else:
    print("   No adjacent states found.")

# Query 3: List coastal states adjacent to Mississippi
# Logic: Find all x where (1) Mississippi is adjacent to x, AND (2) x is a coastal state
query3_result = run(0, x, adjacent('Mississippi', x), coastal(x))
print("\n3. List of coastal states adjacent to Mississippi:")
if query3_result:
    for idx, state in enumerate(query3_result, 1):
        print(f"   {idx}. {state}")
else:
    print("   No coastal adjacent states found.")

# Query 4: List 7 states that border a coastal state (FIXED ERROR HERE!)
# Logic: Find all x where (1) y is a coastal state, AND (2) y is adjacent to x
# Fix: Pass multiple goals as separate arguments (not a tuple) to run()
# Use set() to remove duplicates (e.g., a state bordering 2 coastal states won't repeat)
query4_raw = run(0, x, coastal(y), adjacent(y, x))  # No tuple! Separate goals.
query4_unique = list(set(query4_raw))  # Deduplicate results
query4_result = query4_unique[:7]  # Limit to first 7 states
print("\n4. List of 7 states that border a coastal state:")
if query4_result:
    for idx, state in enumerate(query4_result, 1):
        print(f"   {idx}. {state}")
else:
    print("   No states found that border a coastal state.")

# Query 5: List states adjacent to both Arkansas and Kentucky
# Logic: Find all x where (1) Arkansas is adjacent to x, AND (2) Kentucky is adjacent to x
query5_result = run(0, x, adjacent('Arkansas', x), adjacent('Kentucky', x))
print("\n5. List of states adjacent to both Arkansas and Kentucky:")
if query5_result:
    for idx, state in enumerate(query5_result, 1):
        print(f"   {idx}. {state}")
else:
    print("   No states found adjacent to both Arkansas and Kentucky.")

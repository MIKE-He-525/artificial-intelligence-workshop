from simpleai.search import CspProblem, backtrack

def constraint_func(names, values):
    """Constraint: Adjacent regions must have different colors."""
    return values[0] != values[1]

def test_coloring(num_colors):
    """
    Test if the region-coloring problem can be solved with a given number of colors.
    Args: num_colors (int): Number of colors to test (2, 3, or 4).
    Returns: Solution dict if solvable, None otherwise.
    """
    # Step 1: Define all regions (variables in CSP)
    regions = (
        'Mark', 'Julia', 'Steve', 'Amanda', 
        'Brian', 'Joanne', 'Derek', 'Allan', 
        'Michelle', 'Kelly'
    )
    
    # Step 2: Define color sets based on the number of colors to test
    color_options = {
        2: ['red', 'green'],
        3: ['red', 'green', 'blue'],
        4: ['red', 'green', 'blue', 'gray']
    }[num_colors]  # Select color set for testing
    
    # Step 3: Assign color options to each region (domain in CSP)
    domains = {region: color_options for region in regions}
    
    # Step 4: Define adjacency constraints (all adjacent pairs)
    constraints = [
        (('Mark', 'Julia'), constraint_func),
        (('Mark', 'Steve'), constraint_func),
        (('Julia', 'Steve'), constraint_func),
        (('Julia', 'Amanda'), constraint_func),
        (('Julia', 'Brian'), constraint_func),
        (('Julia', 'Derek'), constraint_func),
        (('Steve', 'Amanda'), constraint_func),
        (('Steve', 'Michelle'), constraint_func),
        (('Steve', 'Allan'), constraint_func),
        (('Amanda', 'Michelle'), constraint_func),
        (('Amanda', 'Derek'), constraint_func),
        (('Brian', 'Derek'), constraint_func),
        (('Brian', 'Kelly'), constraint_func),
        (('Joanne', 'Michelle'), constraint_func),
        (('Joanne', 'Amanda'), constraint_func),
        (('Joanne', 'Derek'), constraint_func),
        (('Derek', 'Kelly'), constraint_func),
        (('Joanne', 'Kelly'), constraint_func)
    ]
    
    # Step 5: Initialize CSP problem and solve with backtracking
    problem = CspProblem(regions, domains, constraints)
    solution = backtrack(problem)  # Backtracking finds a valid solution if it exists
    
    return solution

if __name__ == '__main__':
    # Test color counts from 2 to 4 (ascending order to find the minimum)
    for color_count in [2, 3, 4]:
        print(f"=== Testing {color_count} colors ===")
        result = test_coloring(color_count)
        
        if result:
            # Print valid solution if found
            print(f"[OK] {color_count} colors are sufficient! Solution:")
            for region, color in sorted(result.items()):
                print(f"  {region} → {color}")
            # Since we test in ascending order, the first valid count is optimal
            print(f"\n Optimal solution: {color_count} colors (no need to test more colors).")
            break  # Exit loop early (no need to test larger color sets)
        else:
            # No solution found with current color count
            print(f" {color_count} colors are insufficient (no valid solution).\n")
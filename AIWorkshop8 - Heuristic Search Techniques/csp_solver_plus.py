from simpleai.search import (
    CspProblem,
    backtrack,
    min_conflicts,
    MOST_CONSTRAINED_VARIABLE,
    HIGHEST_DEGREE_VARIABLE,
    LEAST_CONSTRAINING_VALUE
)

def constraint_unique(variables, values):
    return len(values) == len(set(values))

def constraint_bigger(variables, values):
    return values[0] > values[1]

def constraint_odd_even(variables, values):
    if values[0] % 2 == 0:
        return values[1] % 2 == 1
    else:
        return values[1] % 2 == 0

def constraint_patricia_double_anna(variables, values):
    anna_val, patricia_val = values
    return patricia_val == 2 * anna_val

def constraint_patricia_max(variables, values):
    john_val, anna_val, tom_val, patricia_val = values
    return (patricia_val > john_val and 
            patricia_val > anna_val and 
            patricia_val > tom_val)

if __name__ == '__main__':
    variables = ('John', 'Anna', 'Tom', 'Patricia')
    
    domains = {
        'John': [1, 2, 3, 4],       
        'Anna': [1, 2, 3],          
        'Tom': [2, 3, 4, 5],        
        'Patricia': [2, 4, 6]       
    }
    
   
    constraints = [
        (('John', 'Anna', 'Tom'), constraint_unique),          
        (('Tom', 'Anna'), constraint_bigger),                  
        (('John', 'Patricia'), constraint_odd_even),           
        (('Anna', 'Patricia'), constraint_patricia_double_anna),
        (('John', 'Anna', 'Tom', 'Patricia'), constraint_patricia_max)
    ]
    
    problem = CspProblem(variables, domains, constraints)
    
  
    print("=== CSP Solver (Plus Version) Results ===")
    print("\n1. Normal Backtrack (No Heuristic):")
    print(backtrack(problem))
    
    print("\n2. Backtrack (Most Constrained Variable):")
    print(backtrack(problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE))
    
    print("\n3. Backtrack (Highest Degree Variable):")
    print(backtrack(problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE))
    
    print("\n4. Backtrack (Least Constraining Value):")
    print(backtrack(problem, value_heuristic=LEAST_CONSTRAINING_VALUE))
    
    print("\n5. Minimum Conflicts Heuristic:")
    print(min_conflicts(problem))
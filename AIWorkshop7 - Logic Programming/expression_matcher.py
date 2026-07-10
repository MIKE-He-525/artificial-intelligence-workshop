from kanren import run, var, fact
from kanren.assoccomm import eq_assoccomm as la
from kanren.assoccomm import commutative, associative

# Define mathematical operations
add = 'addition'
mul = 'multiplication'
sub = 'substraction'
div = 'division'

# Both addition and multiplication are communicative processes. 
# Hence, we need to specify it and this can be done as follows
fact(commutative, mul)
fact(commutative, add)
fact(associative, mul)
fact(associative, add)


# Define some variables
a, b, c, d= var('a'), var('b'), var('c'), var('d')

# Generate expressions
expression_orig = (add, (mul, 3, -2), (mul, (add, 1, (mul, 2, 3)), -1))
expression1 = (add, (mul, (add, 1, (mul, 2, a)), b), (mul, 3, c))
expression2 = (add, (mul, c, 3), (mul, b, (add, (mul, 2, a), 1)))
expression3 = (add, (add, (mul, (mul, 2, a), b), b), (mul, 3, c)) 

# Compare expressions
print(run(0, (a, b, c), la(expression1, expression_orig)))
print(run(0, (a, b, c), la(expression2, expression_orig)))
print(run(0, (a, b, c), la(expression3, expression_orig)))  

expression_orig = (sub, 
                  (div, 3, -2), 
                  (div, (add, 3, (mul, 7, 3)), -18)
                 )

expression4 = (sub, 
              (div, a, b), 
              (div, (add, 3, (mul, c, d)), -18)
             )

expression5 = (sub, 
              (div, b, a),
              (div, (add, (mul, d, c), 3), -18) 
             )

expression6 = (sub, 
              (div, (add, 3, (mul, 7, 3)), -18), 
              (div, 3, -2) 
             )

# Compare expressions
print(run(0, (a, b, c, d), la(expression4, expression_orig)))
print(run(0, (a, b, c, d), la(expression5, expression_orig)))
print(run(0, (a, b, c, d), la(expression6, expression_orig)))
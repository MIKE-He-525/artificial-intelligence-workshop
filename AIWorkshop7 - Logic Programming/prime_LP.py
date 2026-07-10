from kanren import isvar, run, membero
from kanren.core import success, fail, condeseq, eq, var
from sympy.ntheory.generate import primerange, isprime

original_list = [23, 4, 27, 17, 13, 10, 21, 29, 3, 32, 11, 19]
MAX_VAL = max(original_list)
# Only consider primes up to the largest value in the list (avoids infinite search)
CANDIDATE_PRIMES = list(primerange(2, MAX_VAL + 1))


def prime_check(x):
    if isvar(x):
        return condeseq([(eq, x, p)] for p in CANDIDATE_PRIMES)
    return success if isprime(x) else fail


x = var()
# Try primes first, then check list membership (correct goal order for kanren)
prime_sublist = run(0, x, prime_check(x), membero(x, original_list))

print("Original list:", original_list)
print("Sublist of prime numbers:", prime_sublist)

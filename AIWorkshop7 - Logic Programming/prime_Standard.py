def generate_primes(n):
    """Generate the first 'n' prime numbers using traditional loops."""
    primes = []
    candidate = 2  # Start checking from the smallest prime
    while len(primes) < n:
        is_prime = True
        # Check divisibility up to sqrt(candidate) for efficiency
        for i in range(2, int(candidate ** 0.5) + 1):
            if candidate % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1  # Check next number
    return primes

# Example: Generate first 7 primes
if __name__ == "__main__":
    num_primes = 7
    primes = generate_primes(num_primes)
    print(f"First {num_primes} primes (Standard): {primes}")
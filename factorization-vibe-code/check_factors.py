#!/usr/bin/env python3

import math

def find_factors(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i, n // i
    return None

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

test_numbers = [176399, 1048573, 16777181]
for n in test_numbers:
    factors = find_factors(n)
    if factors:
        print(f'{n:,} = {factors[0]:,} × {factors[1]:,}')
        # Verify
        print(f'  Verification: {factors[0]} × {factors[1]} = {factors[0] * factors[1]}')
    else:
        if is_prime(n):
            print(f'{n:,} is prime')
        else:
            print(f'{n:,} has no small factors found')

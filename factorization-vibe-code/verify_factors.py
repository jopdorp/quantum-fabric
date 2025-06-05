#!/usr/bin/env python3

print('Verifying 16777181 = 43 × 390167:')
result = 43 * 390167
print(f'43 × 390167 = {result}')
print(f'Does 43 divide 16777181? {16777181 % 43 == 0}')
print(f'16777181 ÷ 43 = {16777181 // 43}')

print()
print('Actual factorization 16777181 = 17 × 986893:')
print(f'17 × 986893 = {17 * 986893}')
print(f'Does 17 divide 16777181? {16777181 % 17 == 0}')

print()
print('Is 43 actually a factor?')
print(f'16777181 % 43 = {16777181 % 43}')

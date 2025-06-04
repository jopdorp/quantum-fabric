#!/usr/bin/env python3
"""
Wave-Based Integer Factorization: Interactive Demo

This script provides an interactive demonstration of the wave-based factorization
approach, allowing users to test the algorithm on various numbers and see
detailed performance metrics.
"""

import time
from math import log2
from enhanced_wave_factorization import EnhancedWaveFactorizer, generate_rsa_like_number, is_prime

def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"

def demo_single_number(N, expected_factors=None, verbose=True):
    """Demonstrate factorization of a single number."""
    print(f"\n{'='*60}")
    print(f"🎯 FACTORIZING N = {N}")
    print(f"{'='*60}")
    
    bit_length = int(log2(N)) + 1
    print(f"Bit length: {bit_length}")
    
    if expected_factors:
        p, q = expected_factors
        print(f"Expected: {p} × {q}")
    
    print(f"Theoretical complexity: O(n²) = O({bit_length}²) = O({bit_length**2})")
    print()
    
    # Create factorizer
    factorizer = EnhancedWaveFactorizer(
        signal_bits=64,
        max_bases=min(80, bit_length * 3),
        verbose=verbose
    )
    
    # Time the factorization
    print("🌊 Starting wave-based factorization...")
    start_time = time.time()
    
    factor = factorizer.wave_factor(N)
    
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\n⏱️  Execution time: {format_time(elapsed)}")
    
    if factor and N % factor == 0 and 1 < factor < N:
        other_factor = N // factor
        print(f"✅ SUCCESS: {N} = {factor} × {other_factor}")
        
        # Verify correctness
        if expected_factors:
            p, q = expected_factors
            if (factor == p and other_factor == q) or (factor == q and other_factor == p):
                print("🎯 Factors match expected values!")
            else:
                print("⚠️  Factors are correct but different from expected")
        
        # Check if factors are prime
        factor_prime = is_prime(factor)
        other_prime = is_prime(other_factor)
        print(f"Factor primality: {factor} is {'prime' if factor_prime else 'composite'}, {other_factor} is {'prime' if other_prime else 'composite'}")
        
    else:
        print(f"❌ FACTORIZATION FAILED")
        if factor:
            print(f"   Returned: {factor} (invalid factor)")
        else:
            print(f"   No factor found")
    
    # Display algorithm statistics
    stats = factorizer.get_statistics()
    print(f"\n📊 Algorithm Statistics:")
    print(f"   • Wave propagation steps: {stats['total_steps']:,}")
    print(f"   • Bases tried: {stats['bases_tried']}")
    print(f"   • Collision detections: Low: {stats['collisions_low']}, Med: {stats['collisions_med']}, High: {stats['collisions_high']}")
    print(f"   • Periods found: {stats['periods_found']}")
    print(f"   • Factors extracted: {stats['factors_extracted']}")
    print(f"   • Trivial factors: {stats['trivial_factors']}")
    
    return factor is not None and N % factor == 0 and 1 < factor < N

def demo_progression():
    """Demonstrate factorization on a progression of increasing difficulty."""
    print("🌊 Wave-Based Integer Factorization: Progressive Demo")
    print("=" * 60)
    print()
    print("This demo shows the algorithm's performance on numbers of increasing size.")
    print("Each test demonstrates different aspects of the wave-based approach.")
    print()
    
    test_cases = [
        # Small composites - should be very fast
        (77, (7, 11), "Small composite (trivial factors expected)"),
        (143, (11, 13), "Small composite (close prime factors)"),
        (323, (17, 19), "Medium composite (twin primes)"),
        
        # Medium composites - testing period detection
        (1147, (31, 37), "Medium composite (period detection test)"),
        (2491, (47, 53), "Medium composite (larger primes)"),
        
        # Larger composites - challenging cases
        (6557, (79, 83), "Large composite (13-bit challenge)"),
        (10403, (101, 103), "Large composite (twin primes)"),
        
        # Generated RSA-like numbers
        (None, None, "16-bit RSA-like number (generated)"),
        (None, None, "18-bit RSA-like number (generated)"),
    ]
    
    successes = 0
    total_time = 0
    
    for i, (N, expected, description) in enumerate(test_cases, 1):
        # Generate RSA-like numbers for None cases
        if N is None:
            bits = 16 if "16-bit" in description else 18
            N, p, q = generate_rsa_like_number(bits)
            expected = (p, q)
        
        print(f"\n🔍 Test {i}: {description}")
        print("-" * 40)
        
        start_time = time.time()
        success = demo_single_number(N, expected, verbose=False)
        elapsed = time.time() - start_time
        
        total_time += elapsed
        if success:
            successes += 1
        
        print(f"\nResult: {'✅ SUCCESS' if success else '❌ FAILED'} in {format_time(elapsed)}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📈 DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"Success rate: {successes}/{len(test_cases)} ({100*successes/len(test_cases):.1f}%)")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time: {format_time(total_time/len(test_cases))}")

def interactive_mode():
    """Interactive mode for user input."""
    print("\n🎮 Interactive Mode")
    print("=" * 30)
    print("Enter numbers to factorize (or 'quit' to exit)")
    print("Examples: 77, 143, 1147, 6557")
    print()
    
    while True:
        try:
            user_input = input("Enter number to factorize: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            N = int(user_input)
            
            if N < 4:
                print("Please enter a number ≥ 4")
                continue
            
            if is_prime(N):
                print(f"{N} is prime - no factorization needed!")
                continue
            
            demo_single_number(N, verbose=True)
            
        except ValueError:
            print("Please enter a valid integer")
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break

def main():
    """Main demo function."""
    print("🌊 Wave-Based Integer Factorization: Interactive Demo")
    print("=" * 60)
    print()
    print("This demo showcases the enhanced wave-based factorization algorithm")
    print("with polynomial-time complexity O(n²) to O(n³) where n = log₂(N).")
    print()
    print("Choose a demo mode:")
    print("1. Progressive demonstration (automated test suite)")
    print("2. Interactive mode (enter your own numbers)")
    print("3. Quick showcase (single impressive example)")
    print()
    
    while True:
        try:
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == '1':
                demo_progression()
                break
            elif choice == '2':
                interactive_mode()
                break
            elif choice == '3':
                # Quick showcase with a challenging number
                print("\n🚀 Quick Showcase: 13-bit Challenge")
                demo_single_number(6557, (79, 83), verbose=True)
                break
            else:
                print("Please enter 1, 2, or 3")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break

if __name__ == "__main__":
    main()

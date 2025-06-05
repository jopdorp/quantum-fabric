#!/usr/bin/env python3
"""
Wave Factorization Analysis Tool

This tool provides comprehensive analysis and comparison between different
wave-based factorization approaches, including performance metrics,
algorithm behavior analysis, and theoretical complexity validation.
"""

import time
import random
import statistics
from math import log2, gcd
from typing import List, Dict, Tuple, Optional

# Import our factorization implementations
from enhanced_wave_factorization import EnhancedWaveFactorizer, generate_rsa_like_number, is_prime

class WaveAnalyzer:
    """Comprehensive analysis tool for wave-based factorization algorithms."""
    
    def __init__(self):
        self.results = []
        
    def generate_test_suite(self, bit_ranges: List[Tuple[int, int]], count_per_range: int = 5) -> List[Tuple[int, int, int]]:
        """Generate a comprehensive test suite of semiprimes."""
        test_cases = []
        
        for min_bits, max_bits in bit_ranges:
            for _ in range(count_per_range):
                bits = random.randint(min_bits, max_bits)
                N, p, q = generate_rsa_like_number(bits)
                test_cases.append((N, p, q))
        
        # Sort by number size for consistent testing
        test_cases.sort(key=lambda x: x[0])
        return test_cases
    
    def benchmark_algorithm(self, algorithm_name: str, factorizer_class, test_cases: List[Tuple[int, int, int]], **kwargs) -> Dict:
        """Benchmark a factorization algorithm on test cases."""
        results = {
            'algorithm': algorithm_name,
            'successes': 0,
            'failures': 0,
            'times': [],
            'bit_lengths': [],
            'success_by_bits': {},
            'detailed_results': []
        }
        
        for N, expected_p, expected_q in test_cases:
            bit_length = int(log2(N)) + 1
            
            # Create factorizer instance
            factorizer = factorizer_class(**kwargs)
            
            # Time the factorization
            start_time = time.time()
            factor = factorizer.wave_factor(N)
            elapsed = time.time() - start_time
            
            # Check result
            success = factor is not None and N % factor == 0 and 1 < factor < N
            
            if success:
                results['successes'] += 1
                other_factor = N // factor
                correct_factors = (factor == expected_p and other_factor == expected_q) or \
                                (factor == expected_q and other_factor == expected_p)
            else:
                correct_factors = False
            
            # Record detailed result
            result_detail = {
                'N': N,
                'expected_factors': (expected_p, expected_q),
                'found_factor': factor,
                'other_factor': N // factor if factor else None,
                'bit_length': bit_length,
                'time': elapsed,
                'success': success,
                'correct_factors': correct_factors,
                'stats': factorizer.get_statistics() if hasattr(factorizer, 'get_statistics') else {}
            }
            
            results['detailed_results'].append(result_detail)
            results['times'].append(elapsed)
            results['bit_lengths'].append(bit_length)
            
            # Track success by bit length
            if bit_length not in results['success_by_bits']:
                results['success_by_bits'][bit_length] = {'total': 0, 'success': 0}
            results['success_by_bits'][bit_length]['total'] += 1
            if success:
                results['success_by_bits'][bit_length]['success'] += 1
            
            if not success:
                results['failures'] += 1
        
        # Calculate summary statistics
        results['total_tests'] = len(test_cases)
        results['success_rate'] = results['successes'] / results['total_tests'] if results['total_tests'] > 0 else 0
        results['avg_time'] = statistics.mean(results['times']) if results['times'] else 0
        results['median_time'] = statistics.median(results['times']) if results['times'] else 0
        results['total_time'] = sum(results['times'])
        
        return results
    
    def complexity_analysis(self, results: Dict) -> Dict:
        """Analyze the complexity characteristics of the algorithm."""
        analysis = {
            'theoretical_complexity': 'O(n²) to O(n³)',
            'observed_scaling': {},
            'time_vs_bits': [],
            'steps_vs_bits': []
        }
        
        # Group results by bit length
        bit_groups = {}
        for result in results['detailed_results']:
            bit_len = result['bit_length']
            if bit_len not in bit_groups:
                bit_groups[bit_len] = []
            bit_groups[bit_len].append(result)
        
        # Analyze scaling for each bit length
        for bit_len, group in bit_groups.items():
            if len(group) > 0:
                avg_time = statistics.mean([r['time'] for r in group])
                avg_steps = statistics.mean([r['stats'].get('total_steps', 0) for r in group if r['stats']])
                
                analysis['time_vs_bits'].append((bit_len, avg_time))
                analysis['steps_vs_bits'].append((bit_len, avg_steps))
                
                # Theoretical vs observed
                theoretical_ops = bit_len ** 2  # O(n²)
                analysis['observed_scaling'][bit_len] = {
                    'avg_time': avg_time,
                    'avg_steps': avg_steps,
                    'theoretical_ops': theoretical_ops,
                    'efficiency_ratio': avg_steps / theoretical_ops if theoretical_ops > 0 else 0
                }
        
        return analysis
    
    def collision_analysis(self, results: Dict) -> Dict:
        """Analyze collision detection effectiveness."""
        analysis = {
            'total_collisions': {'low': 0, 'med': 0, 'high': 0},
            'collision_rates': {},
            'collision_effectiveness': {}
        }
        
        for result in results['detailed_results']:
            stats = result.get('stats', {})
            bit_len = result['bit_length']
            
            if bit_len not in analysis['collision_rates']:
                analysis['collision_rates'][bit_len] = {'low': [], 'med': [], 'high': []}
            
            # Collect collision data
            for res_type in ['low', 'med', 'high']:
                collisions = stats.get(f'collisions_{res_type}', 0)
                analysis['total_collisions'][res_type] += collisions
                analysis['collision_rates'][bit_len][res_type].append(collisions)
        
        # Calculate effectiveness metrics
        for bit_len, rates in analysis['collision_rates'].items():
            analysis['collision_effectiveness'][bit_len] = {}
            for res_type in ['low', 'med', 'high']:
                if rates[res_type]:
                    analysis['collision_effectiveness'][bit_len][res_type] = {
                        'avg_collisions': statistics.mean(rates[res_type]),
                        'max_collisions': max(rates[res_type]),
                        'collision_rate': sum(rates[res_type]) / len(rates[res_type])
                    }
        
        return analysis
    
    def generate_report(self, results: Dict, complexity_analysis: Dict, collision_analysis: Dict) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append(f"WAVE FACTORIZATION ANALYSIS REPORT")
        report.append(f"Algorithm: {results['algorithm']}")
        report.append("=" * 80)
        report.append("")
        
        # Performance Summary
        report.append("📊 PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total tests: {results['total_tests']}")
        report.append(f"Successes: {results['successes']}")
        report.append(f"Failures: {results['failures']}")
        report.append(f"Success rate: {results['success_rate']:.1%}")
        report.append(f"Average time: {results['avg_time']:.4f}s")
        report.append(f"Median time: {results['median_time']:.4f}s")
        report.append(f"Total time: {results['total_time']:.4f}s")
        report.append("")
        
        # Success by Bit Length
        report.append("🎯 SUCCESS RATE BY BIT LENGTH")
        report.append("-" * 40)
        for bit_len in sorted(results['success_by_bits'].keys()):
            data = results['success_by_bits'][bit_len]
            rate = data['success'] / data['total'] if data['total'] > 0 else 0
            report.append(f"{bit_len:2d} bits: {data['success']:2d}/{data['total']:2d} ({rate:.1%})")
        report.append("")
        
        # Complexity Analysis
        report.append("⚡ COMPLEXITY ANALYSIS")
        report.append("-" * 40)
        report.append(f"Theoretical: {complexity_analysis['theoretical_complexity']}")
        report.append("")
        report.append("Bit Length | Avg Time | Avg Steps | Theoretical | Efficiency")
        report.append("-" * 60)
        for bit_len in sorted(complexity_analysis['observed_scaling'].keys()):
            data = complexity_analysis['observed_scaling'][bit_len]
            report.append(f"{bit_len:10d} | {data['avg_time']:8.4f} | {data['avg_steps']:9.0f} | {data['theoretical_ops']:11.0f} | {data['efficiency_ratio']:10.2f}")
        report.append("")
        
        # Collision Analysis
        report.append("💥 COLLISION DETECTION ANALYSIS")
        report.append("-" * 40)
        total_low = collision_analysis['total_collisions']['low']
        total_med = collision_analysis['total_collisions']['med']
        total_high = collision_analysis['total_collisions']['high']
        report.append(f"Total collisions - Low: {total_low}, Medium: {total_med}, High: {total_high}")
        report.append("")
        
        if collision_analysis['collision_effectiveness']:
            report.append("Collision effectiveness by bit length:")
            for bit_len in sorted(collision_analysis['collision_effectiveness'].keys()):
                data = collision_analysis['collision_effectiveness'][bit_len]
                report.append(f"{bit_len} bits:")
                for res_type in ['low', 'med', 'high']:
                    if res_type in data:
                        eff = data[res_type]
                        report.append(f"  {res_type:>6}: avg={eff['avg_collisions']:.1f}, max={eff['max_collisions']}")
        
        report.append("")
        
        # Detailed Failures
        failures = [r for r in results['detailed_results'] if not r['success']]
        if failures:
            report.append("❌ DETAILED FAILURE ANALYSIS")
            report.append("-" * 40)
            for failure in failures[:10]:  # Show first 10 failures
                report.append(f"N = {failure['N']} ({failure['bit_length']} bits)")
                report.append(f"  Expected: {failure['expected_factors'][0]} × {failure['expected_factors'][1]}")
                report.append(f"  Time: {failure['time']:.4f}s")
                if failure['stats']:
                    stats = failure['stats']
                    report.append(f"  Steps: {stats.get('total_steps', 0)}, Periods: {stats.get('periods_found', 0)}")
                report.append("")
        
        return "\n".join(report)
    
    def run_comprehensive_analysis(self):
        """Run a comprehensive analysis of the enhanced wave factorizer."""
        print("🌊 Starting Comprehensive Wave Factorization Analysis")
        print("=" * 60)
        
        # Generate test suite
        print("Generating test suite...")
        test_cases = self.generate_test_suite([
            (8, 10),   # Small numbers
            (11, 13),  # Medium numbers  
            (14, 16),  # Larger numbers
            (17, 19),  # Challenge numbers
        ], count_per_range=8)
        
        print(f"Generated {len(test_cases)} test cases")
        print()
        
        # Benchmark enhanced algorithm
        print("Benchmarking Enhanced Wave Factorizer...")
        enhanced_results = self.benchmark_algorithm(
            "Enhanced Wave Factorizer",
            EnhancedWaveFactorizer,
            test_cases,
            signal_bits=48,
            max_bases=60,
            verbose=False
        )
        
        # Perform analyses
        complexity_analysis = self.complexity_analysis(enhanced_results)
        collision_analysis = self.collision_analysis(enhanced_results)
        
        # Generate and display report
        report = self.generate_report(enhanced_results, complexity_analysis, collision_analysis)
        print(report)
        
        # Save results
        with open('wave_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n📄 Full report saved to 'wave_analysis_report.txt'")
        
        return enhanced_results, complexity_analysis, collision_analysis

def main():
    """Run the comprehensive wave factorization analysis."""
    analyzer = WaveAnalyzer()
    results, complexity, collisions = analyzer.run_comprehensive_analysis()
    
    # Additional quick tests on specific challenging numbers
    print("\n" + "=" * 60)
    print("🎯 CHALLENGE TESTS")
    print("=" * 60)
    
    challenge_numbers = [
        (6557, 79, 83),    # 13-bit
        (10403, 101, 103), # 14-bit  
        (16637, 127, 131), # 15-bit
        (22499, 149, 151), # 15-bit
        (32041, 179, 179), # 15-bit (square)
    ]
    
    for N, p, q in challenge_numbers:
        print(f"\nChallenge: N = {N} = {p} × {q} ({int(log2(N))+1} bits)")
        
        factorizer = EnhancedWaveFactorizer(signal_bits=64, max_bases=80, verbose=False)
        
        start_time = time.time()
        factor = factorizer.wave_factor(N)
        elapsed = time.time() - start_time
        
        if factor and N % factor == 0:
            other = N // factor
            print(f"✅ SUCCESS: {factor} × {other} in {elapsed:.4f}s")
            
            stats = factorizer.get_statistics()
            print(f"   Steps: {stats['total_steps']}, Collisions: {stats['collisions_low']+stats['collisions_med']+stats['collisions_high']}")
        else:
            print(f"❌ FAILED in {elapsed:.4f}s")

if __name__ == "__main__":
    main()

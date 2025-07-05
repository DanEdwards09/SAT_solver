#!/usr/bin/env python3
"""
Script to establish baseline performance for your current DPLL solver
"""
import sys
import os
import json
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.solver.dpll_solver import DPLLSolver
from src.benchmarks.benchmark_runner import BenchmarkRunner
from src.benchmarks.test_suites import get_baseline_test_suite, get_quick_test_suite, get_performance_target_suite


def main():
    print("=" * 70)
    print("BASELINE PERFORMANCE ESTABLISHMENT")
    print("=" * 70)
    print("This script will test your current DPLL solver to establish")
    print("baseline performance metrics for comparison with improvements.")
    print()
    
    # Ask user which test suite to run
    print("Available test suites:")
    print("1. Quick test (5 small problems, ~1 minute)")
    print("2. Baseline test (comprehensive, ~15 minutes)")
    print("3. Performance target test (specific to requirements, ~10 minutes)")
    print("4. All tests (may take 30+ minutes)")
    
    choice = input("\nSelect test suite (1-4): ").strip()
    
    if choice == "1":
        test_suite = get_quick_test_suite()
        suite_name = "quick"
        timeout = 60.0  # 1 minute timeout for quick tests
    elif choice == "2":
        test_suite = get_baseline_test_suite()
        suite_name = "baseline"
        timeout = 300.0  # 5 minute timeout
    elif choice == "3":
        test_suite = get_performance_target_suite()
        suite_name = "performance_targets"
        timeout = 300.0  # 5 minute timeout
    elif choice == "4":
        test_suite = get_baseline_test_suite() + get_performance_target_suite()
        suite_name = "comprehensive"
        timeout = 300.0
    else:
        print("Invalid choice. Using quick test suite.")
        test_suite = get_quick_test_suite()
        suite_name = "quick"
        timeout = 60.0
    
    print(f"\nRunning {suite_name} test suite with {len(test_suite)} tests...")
    print(f"Timeout per test: {timeout} seconds")
    print("=" * 70)
    
    # Initialize solver and benchmark runner
    solver = DPLLSolver(verbose=False)
    runner = BenchmarkRunner(timeout=timeout)
    
    # Run benchmarks
    start_time = datetime.now()
    results = runner.run_test_suite(solver, test_suite)
    end_time = datetime.now()
    
    # Print summary
    runner.print_summary(results)
    
    # Analyze specific performance targets
    if choice in ["2", "3", "4"]:  # If running performance-related tests
        analyze_performance_targets(results)
    
    # Save results
    save_baseline_results(results, suite_name)
    
    print(f"\nTotal execution time: {(end_time - start_time).total_seconds():.1f} seconds")
    print("\nBaseline testing complete!")


def analyze_performance_targets(results):
    """Analyze results against the specific performance targets from requirements"""
    print("\n" + "=" * 70)
    print("PERFORMANCE TARGET ANALYSIS")
    print("=" * 70)
    print("Comparing results against requirements targets:")
    print("- 50-vertex: 90% solved in 10 seconds")
    print("- 75-vertex: 75% solved in 60 seconds") 
    print("- 100-vertex: 50% solved in 300 seconds")
    print()
    
    # Group results by vertex count
    by_vertex_count = {}
    for result in results:
        if 'vertex_count' in result.graph_info:
            vertex_count = result.graph_info['vertex_count']
            if vertex_count not in by_vertex_count:
                by_vertex_count[vertex_count] = []
            by_vertex_count[vertex_count].append(result)
    
    # Analyze against targets
    targets = [
        (50, 10.0, 0.90),  # 50 vertices, 10 seconds, 90% success rate
        (75, 60.0, 0.75),  # 75 vertices, 60 seconds, 75% success rate  
        (100, 300.0, 0.50) # 100 vertices, 300 seconds, 50% success rate
    ]
    
    for target_vertices, target_time, target_success_rate in targets:
        if target_vertices in by_vertex_count:
            test_results = by_vertex_count[target_vertices]
            
            # Calculate success rate within target time
            within_target = [r for r in test_results 
                           if not r.timed_out and not r.error_message and r.solve_time <= target_time]
            actual_success_rate = len(within_target) / len(test_results)
            
            # Calculate average time for successful tests
            successful = [r for r in test_results if not r.timed_out and not r.error_message]
            avg_time = sum(r.solve_time for r in successful) / len(successful) if successful else float('inf')
            
            print(f"{target_vertices}-vertex problems:")
            print(f"  Target: {target_success_rate*100:.0f}% in {target_time}s")
            print(f"  Actual: {actual_success_rate*100:.1f}% in {target_time}s ({len(within_target)}/{len(test_results)})")
            print(f"  Average solve time: {avg_time:.3f}s")
            
            if actual_success_rate >= target_success_rate:
                print(f"  ✓ MEETS TARGET")
            else:
                shortfall = (target_success_rate - actual_success_rate) * 100
                print(f"  ✗ BELOW TARGET by {shortfall:.1f} percentage points")
            print()
    
    # Overall assessment
    print("BASELINE ASSESSMENT:")
    print("Based on these results, your enhanced solver should aim for:")
    
    for target_vertices, target_time, target_success_rate in targets:
        if target_vertices in by_vertex_count:
            test_results = by_vertex_count[target_vertices]
            within_target = [r for r in test_results 
                           if not r.timed_out and not r.error_message and r.solve_time <= target_time]
            actual_success_rate = len(within_target) / len(test_results)
            
            if actual_success_rate < target_success_rate:
                needed_improvement = target_success_rate / max(actual_success_rate, 0.01)
                print(f"- {needed_improvement:.1f}× improvement on {target_vertices}-vertex problems")


def save_baseline_results(results, suite_name):
    """Save results for future comparison"""
    os.makedirs('results/baseline_performance', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/baseline_performance/baseline_{suite_name}_{timestamp}.json'
    
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_results.append({
            'test_name': result.test_name,
            'graph_info': result.graph_info,
            'cnf_size': result.cnf_size,
            'solve_time': result.solve_time,
            'is_satisfiable': result.is_satisfiable,
            'solution_valid': result.solution_valid,
            'solver_stats': result.solver_stats,
            'timed_out': result.timed_out,
            'error_message': result.error_message
        })
    
    # Save with metadata
    baseline_data = {
        'timestamp': timestamp,
        'suite_name': suite_name,
        'solver_config': 'baseline_dpll',
        'total_tests': len(results),
        'results': json_results
    }
    
    with open(filename, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    print("You can use this file to compare with enhanced solver performance.")


if __name__ == "__main__":
    main()
"""
Benchmark Runner for SAT Solver Performance Testing
"""
import time
import os
import signal
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark test results"""
    test_name: str
    graph_info: Dict[str, Any]
    cnf_size: int
    solve_time: float
    is_satisfiable: bool
    solution_valid: bool
    solver_stats: Dict[str, int]
    timed_out: bool = False
    error_message: str = ""


class TimeoutException(Exception):
    """Exception raised when solver times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Solver timed out")


class BenchmarkRunner:
    """Runs comprehensive benchmarks on SAT solvers"""
    
    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout
        self.results: List[BenchmarkResult] = []
    
    def run_single_test(self, solver, cnf_formula: List[List[int]], 
                       test_name: str, graph_info: Dict[str, Any]) -> BenchmarkResult:
        """
        Run a single benchmark test with comprehensive monitoring
        
        Args:
            solver: The SAT solver instance
            cnf_formula: The CNF formula to solve
            test_name: Name of the test
            graph_info: Information about the graph
            
        Returns:
            BenchmarkResult containing all test metrics
        """
        print(f"  Running: {test_name}")
        
        # Set up timeout (Unix/Mac only)
        timed_out = False
        error_message = ""
        
        start_time = time.time()
        
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))
            
            # Run the solver
            result = solver.solve(cnf_formula)
            end_time = time.time()
            
            # Clear the alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
        except TimeoutException:
            result = (False, {})
            end_time = start_time + self.timeout
            timed_out = True
            error_message = f"Timed out after {self.timeout} seconds"
        except Exception as e:
            result = (False, {})
            end_time = time.time()
            error_message = f"Error during solving: {str(e)}"
        
        solve_time = end_time - start_time
        
        # Validate solution if satisfiable
        solution_valid = True
        if result[0] and 'vertex_list' in graph_info and 'edges' in graph_info:
            from ..graph.encoders import decode_graph_coloring_solution, validate_graph_coloring
            try:
                coloring = decode_graph_coloring_solution(result[1], graph_info['vertex_list'], graph_info['colors'])
                solution_valid = validate_graph_coloring(graph_info['vertex_list'], graph_info['edges'], coloring)
            except Exception:
                solution_valid = False
        
        # Get solver statistics if available
        solver_stats = {}
        if hasattr(solver, 'get_statistics'):
            solver_stats = solver.get_statistics()
        
        return BenchmarkResult(
            test_name=test_name,
            graph_info=graph_info,
            cnf_size=len(cnf_formula),
            solve_time=solve_time,
            is_satisfiable=result[0],
            solution_valid=solution_valid,
            solver_stats=solver_stats,
            timed_out=timed_out,
            error_message=error_message
        )
    
    def run_test_suite(self, solver, test_suite: List[Dict[str, Any]]) -> List[BenchmarkResult]:
        """
        Run a complete test suite
        
        Args:
            solver: The SAT solver instance
            test_suite: List of test configurations
            
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        print(f"\nRunning test suite with {len(test_suite)} tests...")
        print("=" * 70)
        
        for i, test_config in enumerate(test_suite, 1):
            print(f"\nTest {i}/{len(test_suite)}: {test_config['name']}")
            
            try:
                # Generate graph
                graph_gen_func = test_config['graph_generator']
                vertices, edges = graph_gen_func(**test_config['graph_params'])
                
                # Generate CNF
                from ..graph.encoders import create_graph_coloring_cnf
                cnf_formula = create_graph_coloring_cnf(vertices, edges, test_config['colors'])
                
                # Prepare graph info
                graph_info = {
                    'type': test_config['name'],
                    'vertex_count': len(vertices),
                    'edge_count': len(edges),
                    'colors': test_config['colors'],
                    'vertex_list': vertices,
                    'edges': edges,
                    'edge_density': len(edges) / (len(vertices) * (len(vertices) - 1) / 2) if len(vertices) > 1 else 0
                }
                
                print(f"  Graph: {len(vertices)} vertices, {len(edges)} edges, {test_config['colors']} colors")
                print(f"  CNF: {len(cnf_formula)} clauses")
                
                # Run the test
                result = self.run_single_test(solver, cnf_formula, test_config['name'], graph_info)
                results.append(result)
                
                # Print immediate feedback
                if result.timed_out:
                    status = "TIMEOUT"
                elif result.error_message:
                    status = f"ERROR: {result.error_message}"
                elif result.is_satisfiable:
                    status = "SAT" if result.solution_valid else "SAT (invalid)"
                else:
                    status = "UNSAT"
                
                print(f"  Result: {status} in {result.solve_time:.3f}s")
                
                if result.solver_stats:
                    stats = result.solver_stats
                    print(f"  Stats: {stats.get('decisions', 0)} decisions, {stats.get('unit_propagations', 0)} unit props, {stats.get('backtracks', 0)} backtracks")
                
            except Exception as e:
                print(f"  ERROR: Failed to run test - {str(e)}")
                # Create error result
                error_result = BenchmarkResult(
                    test_name=test_config['name'],
                    graph_info={'error': str(e)},
                    cnf_size=0,
                    solve_time=0.0,
                    is_satisfiable=False,
                    solution_valid=False,
                    solver_stats={},
                    timed_out=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print a summary of benchmark results"""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        # Group by vertex count
        by_vertex_count = {}
        for result in results:
            if 'vertex_count' in result.graph_info:
                vertex_count = result.graph_info['vertex_count']
                if vertex_count not in by_vertex_count:
                    by_vertex_count[vertex_count] = []
                by_vertex_count[vertex_count].append(result)
        
        # Print summary statistics
        for vertex_count in sorted(by_vertex_count.keys()):
            test_results = by_vertex_count[vertex_count]
            solved = [r for r in test_results if not r.timed_out and not r.error_message]
            timed_out = [r for r in test_results if r.timed_out]
            errors = [r for r in test_results if r.error_message and not r.timed_out]
            
            success_rate = len(solved) / len(test_results) * 100
            avg_time = sum(r.solve_time for r in solved) / len(solved) if solved else 0
            
            print(f"\n{vertex_count}-vertex problems:")
            print(f"  Total tests: {len(test_results)}")
            print(f"  Solved: {len(solved)} ({success_rate:.1f}%)")
            print(f"  Timed out: {len(timed_out)}")
            print(f"  Errors: {len(errors)}")
            
            if solved:
                print(f"  Average solve time: {avg_time:.3f}s")
                solve_times = [r.solve_time for r in solved]
                print(f"  Time range: {min(solve_times):.3f}s - {max(solve_times):.3f}s")
                
                # SAT/UNSAT breakdown
                sat_results = [r for r in solved if r.is_satisfiable]
                unsat_results = [r for r in solved if not r.is_satisfiable]
                print(f"  SAT: {len(sat_results)}, UNSAT: {len(unsat_results)}")
        
        # Overall statistics
        total_tests = len(results)
        total_solved = len([r for r in results if not r.timed_out and not r.error_message])
        total_time = sum(r.solve_time for r in results)
        
        print(f"\nOverall Statistics:")
        print(f"  Total tests: {total_tests}")
        print(f"  Solved: {total_solved} ({total_solved/total_tests*100:.1f}%)")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per test: {total_time/total_tests:.3f}s")
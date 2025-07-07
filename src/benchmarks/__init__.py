# src/benchmarks/__init__.py
"""
Benchmarking and Testing Framework
"""
from .benchmark_runner import BenchmarkRunner, BenchmarkResult
from .enhanced_benchmarks import ExperimentalEvaluator
from .test_suites import *

__all__ = [
    'BenchmarkRunner',
    'BenchmarkResult', 
    'ExperimentalEvaluator',
    'get_baseline_test_suite',
    'get_quick_test_suite',
    'get_stress_test_suite',
    'get_performance_target_suite',
    'get_comprehensive_test_suite'
]
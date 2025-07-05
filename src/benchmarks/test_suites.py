"""
Predefined test suites for baseline and comprehensive evaluation
"""
from ..graph.generators import (
    generate_random_graph,
    generate_grid_graph,
    generate_complete_graph,
    generate_path_with_branches,
    generate_cycle_graph,
    generate_bipartite_graph,
    generate_star_graph,
    generate_wheel_graph
)


def get_baseline_test_suite():
    """
    Test suite to establish baseline performance
    Progressive difficulty across different graph sizes and types
    """
    return [
        # =============== 25-VERTEX PROBLEMS ===============
        {
            'name': 'Random_25v_sparse_3c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 25, 'edge_probability': 0.15, 'seed': 100},
            'colors': 3
        },
        {
            'name': 'Random_25v_medium_3c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 25, 'edge_probability': 0.3, 'seed': 101},
            'colors': 3
        },
        {
            'name': 'Grid_5x5_4c',
            'graph_generator': generate_grid_graph,
            'graph_params': {'size': 5},  # 25 vertices
            'colors': 4
        },
        
        # =============== 50-VERTEX PROBLEMS ===============
        {
            'name': 'Random_50v_sparse_3c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 50, 'edge_probability': 0.1, 'seed': 200},
            'colors': 3
        },
        {
            'name': 'Random_50v_medium_3c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 50, 'edge_probability': 0.25, 'seed': 201},
            'colors': 3
        },
        {
            'name': 'Random_50v_dense_4c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 50, 'edge_probability': 0.4, 'seed': 202},
            'colors': 4
        },
        {
            'name': 'Path_50v_branches_3c',
            'graph_generator': generate_path_with_branches,
            'graph_params': {'main_path_length': 35, 'branch_count': 5, 'branch_length': 3},
            'colors': 3
        },
        {
            'name': 'Bipartite_50v_3c',
            'graph_generator': generate_bipartite_graph,
            'graph_params': {'left_size': 25, 'right_size': 25, 'edge_probability': 0.3, 'seed': 203},
            'colors': 3
        },
        
        # =============== 75-VERTEX PROBLEMS ===============
        {
            'name': 'Random_75v_sparse_3c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 75, 'edge_probability': 0.08, 'seed': 300},
            'colors': 3
        },
        {
            'name': 'Random_75v_medium_4c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 75, 'edge_probability': 0.2, 'seed': 301},
            'colors': 4
        },
        {
            'name': 'Random_75v_dense_5c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 75, 'edge_probability': 0.35, 'seed': 302},
            'colors': 5
        },
        {
            'name': 'Grid_approx_75v_4c',
            'graph_generator': generate_grid_graph,
            'graph_params': {'size': 9},  # 81 vertices (close to 75)
            'colors': 4
        },
        
        # =============== 100-VERTEX PROBLEMS ===============
        {
            'name': 'Random_100v_sparse_3c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 100, 'edge_probability': 0.06, 'seed': 400},
            'colors': 3
        },
        {
            'name': 'Random_100v_medium_4c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 100, 'edge_probability': 0.15, 'seed': 401},
            'colors': 4
        },
        {
            'name': 'Random_100v_dense_5c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 100, 'edge_probability': 0.25, 'seed': 402},
            'colors': 5
        },
        {
            'name': 'Grid_10x10_4c',
            'graph_generator': generate_grid_graph,
            'graph_params': {'size': 10},  # 100 vertices
            'colors': 4
        },
        {
            'name': 'Path_100v_branches_3c',
            'graph_generator': generate_path_with_branches,
            'graph_params': {'main_path_length': 70, 'branch_count': 10, 'branch_length': 3},
            'colors': 3
        },
    ]


def get_stress_test_suite():
    """
    Challenging test suite to stress-test the solver
    """
    return [
        # Very dense graphs (likely to be hard)
        {
            'name': 'Random_50v_very_dense_4c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 50, 'edge_probability': 0.6, 'seed': 500},
            'colors': 4
        },
        {
            'name': 'Random_75v_very_dense_5c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 75, 'edge_probability': 0.5, 'seed': 501},
            'colors': 5
        },
        
        # Near-complete graphs (very hard)
        {
            'name': 'Random_30v_near_complete_8c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 30, 'edge_probability': 0.8, 'seed': 502},
            'colors': 8
        },
        
        # Complete graphs (should be solvable with n colors)
        {
            'name': 'Complete_20v_20c',
            'graph_generator': generate_complete_graph,
            'graph_params': {'num_vertices': 20},
            'colors': 20
        },
        
        # Wheels (interesting structure)
        {
            'name': 'Wheel_50v_4c',
            'graph_generator': generate_wheel_graph,
            'graph_params': {'num_vertices': 50},
            'colors': 4
        },
        
        # Large bipartite (should be 2-colorable)
        {
            'name': 'Bipartite_100v_2c',
            'graph_generator': generate_bipartite_graph,
            'graph_params': {'left_size': 50, 'right_size': 50, 'edge_probability': 0.3, 'seed': 503},
            'colors': 2
        },
    ]


def get_quick_test_suite():
    """
    Quick test suite for development and debugging
    """
    return [
        {
            'name': 'Small_random_10v_3c',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 10, 'edge_probability': 0.3, 'seed': 1000},
            'colors': 3
        },
        {
            'name': 'Small_grid_3x3_3c',
            'graph_generator': generate_grid_graph,
            'graph_params': {'size': 3},  # 9 vertices
            'colors': 3
        },
        {
            'name': 'Small_cycle_8v_3c',
            'graph_generator': generate_cycle_graph,
            'graph_params': {'num_vertices': 8},
            'colors': 3
        },
        {
            'name': 'Small_star_10v_2c',
            'graph_generator': generate_star_graph,
            'graph_params': {'num_vertices': 10},
            'colors': 2
        },
        {
            'name': 'Small_complete_5v_5c',
            'graph_generator': generate_complete_graph,
            'graph_params': {'num_vertices': 5},
            'colors': 5
        },
    ]


def get_performance_target_suite():
    """
    Test suite specifically designed to evaluate performance targets
    from the requirements (50v/10s, 75v/60s, 100v/300s)
    """
    suite = []
    
    # 50-vertex tests (target: 90% solved in 10 seconds)
    for i in range(10):
        suite.append({
            'name': f'Target_50v_test_{i+1}',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 50, 'edge_probability': 0.2, 'seed': 2000 + i},
            'colors': 3
        })
    
    # 75-vertex tests (target: 75% solved in 60 seconds)
    for i in range(8):
        suite.append({
            'name': f'Target_75v_test_{i+1}',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 75, 'edge_probability': 0.18, 'seed': 2100 + i},
            'colors': 4
        })
    
    # 100-vertex tests (target: 50% solved in 300 seconds)
    for i in range(6):
        suite.append({
            'name': f'Target_100v_test_{i+1}',
            'graph_generator': generate_random_graph,
            'graph_params': {'num_vertices': 100, 'edge_probability': 0.15, 'seed': 2200 + i},
            'colors': 4
        })
    
    return suite


def get_comprehensive_test_suite():
    """
    Comprehensive test suite combining all test types
    """
    suite = []
    suite.extend(get_baseline_test_suite())
    suite.extend(get_stress_test_suite())
    suite.extend(get_performance_target_suite())
    return suite
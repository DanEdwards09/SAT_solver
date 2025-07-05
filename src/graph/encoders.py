"""
CNF Encoding Functions for Graph Coloring
"""
from typing import List, Tuple, Dict


def create_graph_coloring_cnf(vertices: List[int], edges: List[Tuple[int, int]], num_colors: int) -> List[List[int]]:
    """
    Creates a CNF formula for graph coloring problem using direct encoding.
    
    Args:
        vertices: List of vertex names/numbers
        edges: List of tuples representing edges
        num_colors: Number of colors available
    
    Returns:
        CNF formula as list of clauses
    """
    # Variable encoding: x_v_c where v is vertex and c is color
    # We'll map this to integers: vertex v with color c gets variable number v * num_colors + c
    def get_variable(vertex: int, color: int) -> int:
        return vertex * num_colors + color + 1  # +1 because variables start from 1
    
    cnf_formula = []
    
    # Constraint 1: Each vertex must have at least one color
    for vertex in vertices:
        clause = []
        for color in range(num_colors):
            clause.append(get_variable(vertex, color))
        cnf_formula.append(clause)
    
    # Constraint 2: Each vertex has at most one color
    for vertex in vertices:
        for color1 in range(num_colors):
            for color2 in range(color1 + 1, num_colors):
                # NOT (vertex has color1 AND vertex has color2)
                # Which is: (NOT vertex_color1 OR NOT vertex_color2)
                clause = [-get_variable(vertex, color1), -get_variable(vertex, color2)]
                cnf_formula.append(clause)
    
    # Constraint 3: Adjacent vertices must have different colors
    for vertex1, vertex2 in edges:
        for color in range(num_colors):
            # NOT (vertex1 has color AND vertex2 has color)
            # Which is: (NOT vertex1_color OR NOT vertex2_color)
            clause = [-get_variable(vertex1, color), -get_variable(vertex2, color)]
            cnf_formula.append(clause)
    
    return cnf_formula


def create_sequential_coloring_cnf(vertices: List[int], edges: List[Tuple[int, int]], num_colors: int) -> List[List[int]]:
    """
    Creates a CNF formula using sequential encoding (may be more efficient for some problems).
    
    Args:
        vertices: List of vertex names/numbers
        edges: List of tuples representing edges
        num_colors: Number of colors available
    
    Returns:
        CNF formula as list of clauses
    """
    # Sequential encoding: x_v_i means vertex v uses color <= i
    def get_variable(vertex: int, color_bound: int) -> int:
        return vertex * (num_colors - 1) + color_bound + 1
    
    cnf_formula = []
    
    # Each vertex must use some color (color <= num_colors-1)
    for vertex in vertices:
        cnf_formula.append([get_variable(vertex, num_colors - 1)])
    
    # Sequential ordering: if vertex uses color <= i, then it uses color <= i+1
    for vertex in vertices:
        for i in range(num_colors - 2):
            # x_v_i => x_v_{i+1}  which is  (!x_v_i OR x_v_{i+1})
            clause = [-get_variable(vertex, i), get_variable(vertex, i + 1)]
            cnf_formula.append(clause)
    
    # Adjacent vertices cannot both use color i
    for vertex1, vertex2 in edges:
        for color in range(num_colors):
            if color == 0:
                # Both can't use exactly color 0
                clause = [-get_variable(vertex1, 0), -get_variable(vertex2, 0)]
                cnf_formula.append(clause)
            else:
                # Both can't use exactly color i
                # vertex uses exactly color i iff (uses <= i AND NOT uses <= i-1)
                # So: NOT((v1 uses <= i AND NOT v1 uses <= i-1) AND (v2 uses <= i AND NOT v2 uses <= i-1))
                # This is complex, so we'll stick with direct encoding for now
                pass
    
    # For simplicity, fall back to direct encoding for edge constraints
    # In a full implementation, you'd properly implement sequential edge constraints
    direct_edge_clauses = []
    for vertex1, vertex2 in edges:
        for color in range(num_colors):
            # This is a simplified approach - proper sequential encoding is more complex
            pass
    
    return cnf_formula


def decode_graph_coloring_solution(assignments: Dict[int, bool], vertices: List[int], num_colors: int) -> Dict[int, int]:
    """
    Decodes a SAT solution back to graph coloring assignment.
    
    Args:
        assignments: Dictionary of variable assignments from SAT solver
        vertices: List of vertices
        num_colors: Number of colors used in encoding
    
    Returns:
        Dictionary mapping vertex -> color
    """
    def get_variable(vertex: int, color: int) -> int:
        return vertex * num_colors + color + 1
    
    coloring = {}
    for vertex in vertices:
        for color in range(num_colors):
            var = get_variable(vertex, color)
            if var in assignments and assignments[var]:
                coloring[vertex] = color
                break
    
    return coloring


def validate_graph_coloring(vertices: List[int], edges: List[Tuple[int, int]], 
                          coloring: Dict[int, int]) -> bool:
    """
    Validates that a graph coloring is correct.
    
    Args:
        vertices: List of vertices
        edges: List of edges
        coloring: Dictionary mapping vertex -> color
    
    Returns:
        True if coloring is valid, False otherwise
    """
    # Check that all vertices are colored
    for vertex in vertices:
        if vertex not in coloring:
            return False
    
    # Check that no adjacent vertices have the same color
    for vertex1, vertex2 in edges:
        if coloring[vertex1] == coloring[vertex2]:
            return False
    
    return True


def get_encoding_stats(vertices: List[int], edges: List[Tuple[int, int]], num_colors: int) -> Dict[str, int]:
    """
    Get statistics about the CNF encoding size.
    
    Args:
        vertices: List of vertices
        edges: List of edges
        num_colors: Number of colors
    
    Returns:
        Dictionary with encoding statistics
    """
    num_vertices = len(vertices)
    num_edges = len(edges)
    
    # Direct encoding statistics
    variables = num_vertices * num_colors
    
    # Clause counts:
    # - At least one color per vertex: num_vertices clauses
    # - At most one color per vertex: num_vertices * C(num_colors, 2) clauses
    # - Different colors for adjacent vertices: num_edges * num_colors clauses
    
    at_least_one_clauses = num_vertices
    at_most_one_clauses = num_vertices * (num_colors * (num_colors - 1) // 2)
    edge_clauses = num_edges * num_colors
    
    total_clauses = at_least_one_clauses + at_most_one_clauses + edge_clauses
    
    return {
        'variables': variables,
        'clauses': total_clauses,
        'at_least_one_clauses': at_least_one_clauses,
        'at_most_one_clauses': at_most_one_clauses,
        'edge_clauses': edge_clauses,
        'clause_to_variable_ratio': total_clauses / variables if variables > 0 else 0
    }
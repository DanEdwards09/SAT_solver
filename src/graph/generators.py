"""
Graph Generation Functions
"""
import random
from typing import List, Tuple, Optional


def generate_grid_graph(size: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a grid graph of given size.
    
    Args:
        size: Grid will be size x size
    
    Returns:
        Tuple (vertices, edges)
    """
    vertices = []
    edges = []
    
    # Create vertices for grid
    for i in range(size):
        for j in range(size):
            vertices.append(i * size + j)
    
    # Create edges (each cell connected to adjacent cells)
    for i in range(size):
        for j in range(size):
            current = i * size + j
            # Connect to right neighbor
            if j < size - 1:
                right = i * size + (j + 1)
                edges.append((current, right))
            # Connect to bottom neighbor
            if i < size - 1:
                bottom = (i + 1) * size + j
                edges.append((current, bottom))
    
    return vertices, edges


def generate_random_graph(num_vertices: int, edge_probability: float, seed: Optional[int] = 42) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a random graph with given edge probability.
    
    Args:
        num_vertices: Number of vertices
        edge_probability: Probability of edge existing between any two vertices
        seed: Random seed for reproducibility
    
    Returns:
        Tuple (vertices, edges)
    """
    if seed is not None:
        random.seed(seed)
    
    vertices = list(range(num_vertices))
    edges = []
    
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < edge_probability:
                edges.append((i, j))
    
    return vertices, edges


def generate_complete_graph(num_vertices: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a complete graph where every vertex is connected to every other vertex.
    
    Args:
        num_vertices: Number of vertices
    
    Returns:
        Tuple (vertices, edges)
    """
    vertices = list(range(num_vertices))
    edges = []
    
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            edges.append((i, j))
    
    return vertices, edges


def generate_path_with_branches(main_path_length: int, branch_count: int, branch_length: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a path graph with branches - creates interesting constraint patterns.
    
    Args:
        main_path_length: Length of the main path
        branch_count: Number of branches off the main path
        branch_length: Length of each branch
    
    Returns:
        Tuple (vertices, edges)
    """
    vertices = []
    edges = []
    vertex_id = 0
    
    # Create main path
    main_path = []
    for i in range(main_path_length):
        main_path.append(vertex_id)
        vertices.append(vertex_id)
        vertex_id += 1
    
    # Connect main path
    for i in range(len(main_path) - 1):
        edges.append((main_path[i], main_path[i + 1]))
    
    # Add branches at evenly spaced points
    if branch_count > 0:
        branch_positions = [main_path_length // (branch_count + 1) * (i + 1) for i in range(branch_count)]
        
        for pos in branch_positions:
            if pos < len(main_path):
                # Create branch
                branch_root = main_path[pos]
                prev_vertex = branch_root
                for i in range(branch_length):
                    branch_vertex = vertex_id
                    vertices.append(branch_vertex)
                    edges.append((prev_vertex, branch_vertex))
                    prev_vertex = branch_vertex
                    vertex_id += 1
    
    return vertices, edges


def generate_cycle_graph(num_vertices: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a cycle graph (ring).
    
    Args:
        num_vertices: Number of vertices in the cycle
    
    Returns:
        Tuple (vertices, edges)
    """
    vertices = list(range(num_vertices))
    edges = []
    
    for i in range(num_vertices):
        next_vertex = (i + 1) % num_vertices
        edges.append((i, next_vertex))
    
    return vertices, edges


def generate_bipartite_graph(left_size: int, right_size: int, edge_probability: float, seed: Optional[int] = 42) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a random bipartite graph.
    
    Args:
        left_size: Number of vertices in left partition
        right_size: Number of vertices in right partition
        edge_probability: Probability of edge between left and right vertices
        seed: Random seed for reproducibility
    
    Returns:
        Tuple (vertices, edges)
    """
    if seed is not None:
        random.seed(seed)
    
    vertices = list(range(left_size + right_size))
    edges = []
    
    # Connect left partition (0 to left_size-1) to right partition (left_size to left_size+right_size-1)
    for left in range(left_size):
        for right in range(left_size, left_size + right_size):
            if random.random() < edge_probability:
                edges.append((left, right))
    
    return vertices, edges


def generate_star_graph(num_vertices: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a star graph (one central vertex connected to all others).
    
    Args:
        num_vertices: Total number of vertices
    
    Returns:
        Tuple (vertices, edges)
    """
    vertices = list(range(num_vertices))
    edges = []
    
    # Connect vertex 0 (center) to all other vertices
    for i in range(1, num_vertices):
        edges.append((0, i))
    
    return vertices, edges


def generate_wheel_graph(num_vertices: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Generates a wheel graph (cycle with additional central vertex connected to all).
    
    Args:
        num_vertices: Total number of vertices (including center)
    
    Returns:
        Tuple (vertices, edges)
    """
    if num_vertices < 4:
        raise ValueError("Wheel graph needs at least 4 vertices")
    
    vertices = list(range(num_vertices))
    edges = []
    
    # Create cycle among vertices 1 to num_vertices-1
    for i in range(1, num_vertices - 1):
        edges.append((i, i + 1))
    edges.append((num_vertices - 1, 1))  # Close the cycle
    
    # Connect center (vertex 0) to all vertices in the cycle
    for i in range(1, num_vertices):
        edges.append((0, i))
    
    return vertices, edges
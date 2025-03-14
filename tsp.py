import numpy as np
import itertools
import random
import math
import time
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

def generate_tsp(n: int) -> Dict[int, Tuple[int, int]]:
    """
    Generate a TSP problem with n nodes.
    
    Args:
        n: Number of nodes in the TSP problem.
        
    Returns:
        A dictionary where keys are node numbers (1 to n) and values are 2D coordinates 
        as (x, y) tuples with integer values between -100 and 100.
    """
    tsp = {}
    for i in range(1, n+1):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        tsp[i] = (x, y)
    return tsp

def calculate_tsp_distance(tsp: Dict[int, Tuple[int, int]], path: List[int]) -> float:
    """
    Calculate the total distance of a TSP solution.
    
    Args:
        tsp: Dictionary of node numbers to coordinates.
        path: List of node numbers representing the order of visitation.
        
    Returns:
        Total Euclidean distance of the path (including return to starting point).
    """
    total_distance = 0
    for i in range(len(path)):
        current_node = path[i]
        next_node = path[0] if i == len(path) - 1 else path[i + 1]
        
        x1, y1 = tsp[current_node]
        x2, y2 = tsp[next_node]
        
        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    
    return total_distance

def solve_tsp_brute_force(tsp: Dict[int, Tuple[int, int]]) -> Tuple[List[int], float]:
    """
    Solve TSP using appropriate algorithm based on problem size.
    - Brute force for n <= 10
    - Simulated annealing for n > 10 and n <= 25
    
    Args:
        tsp: Dictionary of node numbers to coordinates.
        
    Returns:
        Tuple of (optimal_path, optimal_distance)
    """
    nodes = list(tsp.keys())
    n = len(nodes)
    
    # For small problems, use brute force
    if n <= 10:
        start_node = nodes[0]
        remaining_nodes = nodes[1:]
        
        # Generate all possible permutations of the remaining nodes
        all_permutations = list(itertools.permutations(remaining_nodes))
        
        best_path = None
        best_distance = float('inf')
        
        for perm in all_permutations:
            # Create a complete path by adding the start node at the beginning
            path = [start_node] + list(perm)
            distance = calculate_tsp_distance(tsp, path)
            
            if distance < best_distance:
                best_distance = distance
                best_path = path
        
        return best_path, best_distance
    
    # For medium-sized problems (n <= 25), use simulated annealing
    elif n <= 25:
        return solve_tsp_simulated_annealing(tsp)
    
    # For larger problems, warn user but still attempt simulated annealing
    else:
        print(f"Warning: Problem size n={n} is large. Solution may take time.")
        return solve_tsp_simulated_annealing(tsp)

def solve_tsp_simulated_annealing(tsp: Dict[int, Tuple[int, int]]) -> Tuple[List[int], float]:
    """
    Solve TSP using Simulated Annealing algorithm.
    
    Args:
        tsp: Dictionary of node numbers to coordinates.
        
    Returns:
        Tuple of (best_path, best_distance)
    """
    nodes = list(tsp.keys())
    n = len(nodes)
    
    # Create distance matrix for faster access during annealing
    distance_matrix = np.zeros((n, n))
    node_to_index = {node: i for i, node in enumerate(nodes)}
    
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                x1, y1 = tsp[node_i]
                x2, y2 = tsp[node_j]
                distance_matrix[i, j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Initialize with a random path
    current_path = nodes.copy()
    random.shuffle(current_path)
    
    # Calculate initial solution cost
    current_distance = calculate_tsp_distance(tsp, current_path)
    
    # Initialize best solution
    best_path = current_path.copy()
    best_distance = current_distance
    
    # Simulated annealing parameters - adjusted based on problem size
    temp = 100.0  # Initial temperature
    cooling_rate = 0.995  # Cooling rate
    min_temp = 0.01  # Minimum temperature
    iterations_per_temp = 100 * n  # Iterations at each temperature
    
    # Simulated annealing main loop
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # Generate a neighbor solution by swapping two cities
            i, j = random.sample(range(n), 2)
            new_path = current_path.copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]
            
            # Calculate the new distance
            new_distance = calculate_tsp_distance(tsp, new_path)
            
            # Decide whether to accept the new solution
            delta = new_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_path = new_path
                current_distance = new_distance
                
                # Update best solution if needed
                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance
        
        # Cool down the temperature
        temp *= cooling_rate
    
    return best_path, best_distance

def create_tsp_dataset(sizes: List[int], count_per_size: int) -> Dict:
    """
    Create a dataset of TSP problems and their solutions.
    
    Args:
        sizes: List of problem sizes (number of nodes).
        count_per_size: Number of problems to generate for each size.
        
    Returns:
        Dictionary containing the generated problems and solutions.
    """
    dataset = {}
    
    # Use a flat progress bar for the overall process
    total_problems = len(sizes) * count_per_size
    pbar = tqdm(total=total_problems, desc="Generating TSP Dataset")
    
    for size in sizes:
        dataset[f"size_{size}"] = []
        
        for i in range(count_per_size):
            # Generate a TSP problem
            problem = generate_tsp(size)
            
            # Solve the problem
            start_time = time.time()
            solution_path, solution_distance = solve_tsp_brute_force(problem)
            solve_time = time.time() - start_time
            
            # Store the problem and solution
            problem_data = {
                "problem_id": f"tsp_{size}_{i+1}",
                "size": size,
                "coordinates": {str(k): v for k, v in problem.items()},  # Convert keys to strings for JSON
                "solution": {
                    "path": solution_path,
                    "distance": solution_distance,
                    "solve_time_seconds": solve_time
                }
            }
            
            dataset[f"size_{size}"].append(problem_data)
            pbar.update(1)
            pbar.set_postfix({"size": size, "problem": i+1, "time": f"{solve_time:.2f}s"})
    
    pbar.close()
    return dataset

def serialize_tsp_dataset(dataset, filename="tsp_dataset.json"):
    """
    Serialize the TSP dataset to a JSON file.
    """
    # Convert NumPy and tuples to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    serializable_dataset = convert_to_serializable(dataset)
    
    with open(filename, 'w') as f:
        json.dump(serializable_dataset, f, indent=2)
    
    print(f"Dataset saved to {filename}")

# Generate the dataset
# if __name__ == "__main__":
#     # Set random seed for reproducibility
#     random.seed(42)
#     np.random.seed(42)
    
#     # Define problem sizes and count per size
#     sizes = [5, 10, 15, 20]
#     count_per_size = 25
    
#     print(f"Creating a dataset with {count_per_size} problems for each size: {sizes}")
#     print(f"Total problems: {len(sizes) * count_per_size}")
    
#     # Create and save the dataset
#     dataset = create_tsp_dataset(sizes, count_per_size)
    
#     # Add metadata
#     dataset["metadata"] = {
#         "total_problems": len(sizes) * count_per_size,
#         "sizes": sizes,
#         "count_per_size": count_per_size,
#         "date_generated": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "description": "TSP problems and solutions dataset"
#     }
    
#     # Save the dataset
#     serialize_tsp_dataset(dataset, "tsp_dataset_100_problems.json")
    
#     # Print summary statistics
#     print("\nDataset Summary:")
#     for size in sizes:
#         problems = dataset[f"size_{size}"]
#         avg_time = sum(p["solution"]["solve_time_seconds"] for p in problems) / len(problems)
#         avg_distance = sum(p["solution"]["distance"] for p in problems) / len(problems)
#         print(f"Size {size}: {len(problems)} problems, Avg solve time: {avg_time:.4f}s, Avg distance: {avg_distance:.2f}")
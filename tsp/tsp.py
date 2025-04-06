import numpy as np
import itertools
import os
import random
import math
import time
import json
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import multiprocessing as mp

def generate_tsp(n: int) -> Dict[int, Tuple[int, int]]:
    """
    Generate a TSP problem with n nodes.
    
    Args:
        n: Number of nodes in the TSP problem.
        
    Returns:
        A dictionary where keys are node indices [0, n) and values are 2D coordinates 
        as (x, y) tuples with integer values between -100 and 100.
    """
    tsp = {}
    for i in range(n):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        tsp[i] = (x, y)
    return tsp

def calculate_tsp_distance(tsp: Dict[int, Tuple[int, int]], path: List[int]) -> float:
    """
    Calculate the total distance of a TSP solution.
    
    Args:
        tsp: Dictionary of node indices to coordinates.
        path: List of node indices representing the order of visitation (including return to start node). Ex: 0 -> 2 -> 4 -> 3 -> 1 -> 0
        
    Returns:
        Total Euclidean distance of the path (including return to start node).
    """
    total_distance = 0
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        
        x1, y1 = tsp[current_node]
        x2, y2 = tsp[next_node]
        
        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    
    return total_distance

def generate_tsp_distance_matrix(tsp: Dict[int, Tuple[int, int]]) -> List[List[float]]:
    """
    Calculate the distance matrix for a TSP problem.

    Args:
        tsp: Dictionary of node indices to coordinates
    
    Returns:
        Distance matrix m where m[i][j] represents the Euclidean distance between nodes i and j.
    """
    n = len(tsp)
    m = []
    for i in range(n):
        row = []
        for j in range(n):
            # Calculate Euclidean distance
            x1, y1 = tsp[i]
            x2, y2 = tsp[j]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            row.append(distance)
        m.append(row)
    return m

def solve_tsp(tsp: Dict[int, Tuple[int, int]]) -> Tuple[List[int], float]:
    """
    Solve TSP using appropriate algorithm based on problem size.
    - Brute force for n <= 10
    - Simulated annealing for 10 < n <= 25
    
    Args:
        tsp: Dictionary of node indices to coordinates.
        
    Returns:
        Tuple of (optimal_path, optimal_distance)
    """
    nodes = list(tsp.keys())
    n = len(nodes)
    
    # For small problems, use brute force
    if n <= 10:
        return solve_tsp_brute_force(tsp)
    
    # For medium-sized problems (n <= 25), use simulated annealing
    elif n <= 25:
        return solve_tsp_simulated_annealing(tsp)
    
    # For larger problems, warn user but still attempt simulated annealing
    else:
        print(f"Warning: Problem size n={n} is large. Solution may take time.")
        return solve_tsp_simulated_annealing(tsp)

def solve_tsp_brute_force(tsp: Dict[int, Tuple[int, int]]) -> Tuple[List[int], float]:
    """
    Solve TSP using brute force.
    
    Args:
        tsp: Dictionary of node indices to coordinates.
        
    Returns:
        Tuple of (optimal_path, optimal_distance)
    """
    nodes = list(tsp.keys())
    
    start_node = nodes[0]
    remaining_nodes = nodes[1:]
        
    # Generate all possible permutations of the remaining nodes
    all_permutations = list(itertools.permutations(remaining_nodes))
        
    best_path = None
    best_distance = float('inf')
        
    for perm in all_permutations:
        # Create a complete path by adding the start node to the beginning and to the end
        path = [start_node] + list(perm) + [start_node]
        distance = calculate_tsp_distance(tsp, path)
            
        if distance < best_distance:
            best_distance = distance
            best_path = path
        
    return best_path, best_distance

def solve_tsp_simulated_annealing(tsp: Dict[int, Tuple[int, int]]) -> Tuple[List[int], float]:
    """
    Solve TSP using Simulated Annealing algorithm.
    
    Args:
        tsp: Dictionary of node indices to coordinates.
        
    Returns:
        Tuple of (best_path, best_distance)
    """
    nodes = list(tsp.keys())
    n = len(nodes)
    
    # Initialize with a random path
    current_path = [0]
    remaining_nodes = [node for node in nodes if node != 0]
    random.shuffle(remaining_nodes)
    current_path.extend(remaining_nodes)
    current_path.extend([0])
    
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
            i, j = random.sample(range(1, n), 2)
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

def solve_and_summarize_tsp(problem_data: Dict[str, Any]) -> Dict[str, Any]:
    # Solve the problem
    start_time = time.time()
    solution_path, solution_distance = solve_tsp(problem_data['tsp'])
    solve_time = time.time() - start_time

    # Remove 'tsp' field and add 'solution' info
    del problem_data['tsp']
    problem_data['solution'] = {
        "path": solution_path,
        "distance": round(solution_distance),
        "solve_time_seconds": solve_time
    }
    return problem_data

def open_tsp_dataset(filename: str) -> Dict:
    """
    Open json dataset at given path. If the file does not exist, return empty dict.

    Args:
        filename: File name of json dataset
    
    Returns:
        Dict: The existing dataset, or empty dict
    """

    if os.path.exists(filename):
        with open(filename, 'r') as file:
            dataset = json.load(file)
    else:
        dataset = {}
    
    return dataset

def serialize_tsp_dataset(dataset: Dict, filename: str) -> None:
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

def create_tsp_dataset(sizes: List[int], count_per_size: int, filename: str) -> None:
    """
    Create a dataset of TSP problems and their solutions. Solve TSP problems in parallel.
    
    Args:
        sizes: List of problem sizes (number of nodes).
        count_per_size: Number of problems to generate for each size.
    """
    
    # Note start time
    start_time = time.time()

    # Create tsp dataset one size at a time
    for size in sizes:
        size_key = f"size_{size}"
        problems = []
        for i in tqdm(range(count_per_size)):
            # Generate a TSP problem
            tsp = generate_tsp(size)
            
            # Store the problem and problem metadata
            problem = {
                "problem_id": f"tsp_{size}_{i}",
                "size": size,
                "coordinates": {str(k): v for k, v in tsp.items()},  # Convert keys to strings for JSON
                "tsp": tsp
            }
            problems.append(problem)
        
        # Solve problems in parallel
        num_cores = mp.cpu_count()
        print(f"Solving {size_key} problems in parallel, using {num_cores} cores")
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(solve_and_summarize_tsp, problems)
        
        # Sort results by problem id
        results = sorted(results, key=lambda x: int(x['problem_id'].split("_")[2]))

        # Open dataset
        dataset = open_tsp_dataset(filename)

        # Add new problems
        dataset[size_key] = results

        # Serialize dataset
        serialize_tsp_dataset(dataset, filename)
    
    # Note end time
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Open dataset
    dataset = open_tsp_dataset(filename)

    # Add metadata
    if not "metadata" in dataset:
        dataset["metadata"] = {
            "total_problems": len(sizes) * count_per_size,
            "generation_time": generation_time,
            "sizes": sizes,
            "count_per_size": count_per_size,
            "date_generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "TSP problems and solutions dataset"
        }
    else:
        dataset["metadata"] = {
            "total_problems": dataset["total_problems"] + len(sizes) * count_per_size,
            "generation_time": dataset["total_problems"] + generation_time,
            "sizes": dataset["sizes"] + sizes,
            "count_per_size": count_per_size,
            "date_generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "TSP problems and solutions dataset"
        }

    # Serialize dataset
    serialize_tsp_dataset(dataset, filename)

# Generate the dataset
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define problem sizes and count per size
    sizes = list(range(5, 16))
    count_per_size = 100
    
    print(f"Creating a dataset with {count_per_size} problems for each size: {sizes}")
    print(f"Total problems: {len(sizes) * count_per_size}")
    
    # Create and save the dataset
    create_tsp_dataset(sizes, count_per_size, "tsp_problems_benchmark_dataset.json")

    # Open dataset
    dataset = open_tsp_dataset("tsp_problems_benchmark_dataset.json")
    
    # Print summary statistics
    print("\nDataset Summary:")
    for size in sizes:
        problems = dataset[f"size_{size}"]
        avg_time = sum(p["solution"]["solve_time_seconds"] for p in problems) / len(problems)
        avg_distance = sum(p["solution"]["distance"] for p in problems) / len(problems)
        print(f"Size {size}: {len(problems)} problems, Avg solve time: {avg_time:.4f}s, Avg distance: {avg_distance:.2f}")
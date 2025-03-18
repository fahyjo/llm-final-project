import json
import random
import math
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm

def load_tsp_dataset(filename: str = "tsp_dataset_100_problems.json") -> Dict:
    """
    Load the TSP dataset from a JSON file.
    
    Args:
        filename: Path to the JSON file containing the TSP dataset.
        
    Returns:
        Dictionary containing the TSP dataset.
    """
    with open(filename, 'r') as f:
        dataset = json.load(f)
    return dataset

def create_distance_matrix(coords: Dict[str, List[float]]) -> Dict[Tuple[int, int], float]:
    """
    Create a distance matrix for the given coordinates.
    
    Args:
        coords: Dictionary mapping node IDs to coordinate pairs.
        
    Returns:
        Dictionary mapping node pairs to distances.
    """
    distance_matrix = {}
    nodes = sorted([int(k) for k in coords.keys()])
    
    for i in nodes:
        for j in nodes:
            if i != j:
                x1, y1 = coords[str(i)]
                x2, y2 = coords[str(j)]
                
                # Calculate Euclidean distance
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distance_matrix[(i-1, j-1)] = distance
    
    return distance_matrix

def format_distance_matrix(distance_matrix: Dict[Tuple[int, int], float], problem_size: int) -> str:
    """
    Format the distance matrix for the LLM prompt.
    
    Args:
        distance_matrix: Dictionary mapping node pairs to distances.
        problem_size: Number of nodes in the TSP problem.
        
    Returns:
        Formatted string of the distance matrix.
    """
    nodes = list(range(problem_size))
    matrix_str = "Distance Matrix:\n"
    
    # Header row
    matrix_str += "    " + " ".join(f"{j:4}" for j in nodes) + "\n"
    
    # Rows
    for i in nodes:
        row_str = f"{i:2}: "
        for j in nodes:
            if i == j:
                distance = 0.0
            else:
                distance = distance_matrix.get((i, j), 0.0)
            row_str += f"{distance:4.1f} "
        matrix_str += row_str + "\n"
    
    return matrix_str

def calculate_path_distance(coords: Dict[str, List[float]], path: List[int]) -> float:
    """
    Calculate the total distance of a path through the given coordinates.
    
    Args:
        coords: Dictionary mapping node IDs to coordinate pairs.
        path: List of node IDs representing the order of visitation.
        
    Returns:
        Total Euclidean distance of the path.
    """
    total_distance = 0
    for i in range(len(path)):
        current_node = str(path[i])
        next_node = path[0] if i == len(path) - 1 else path[i + 1]
        
        x1, y1 = coords[str(int(current_node)+1)]
        x2, y2 = coords[str(int(next_node)+1)]
        
        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    
    return total_distance


def generate_random_solution(problem_size: int) -> List[int]:
    """
    Generate a random solution for a TSP problem, ensuring that all solutions start with 0.
    
    Args:
        problem_size: Number of nodes in the TSP problem.
        
    Returns:
        List of node IDs representing a random path through all nodes, starting with 0.
    """
    if problem_size <= 1:
        return [0] * problem_size  # Handles edge case where problem_size is 0 or 1
    
    nodes = list(range(1, problem_size))  # Exclude 0 from shuffling
    random.shuffle(nodes)
    return [0] + nodes

def generate_sample_solutions(coords: Dict[str, List[float]], num_samples: int = 4) -> List[Dict[str, Any]]:
    """
    Generate sample solutions for a TSP problem.
    
    Args:
        coords: Dictionary mapping node IDs to coordinate pairs.
        num_samples: Number of sample solutions to generate.
        
    Returns:
        List of dictionaries containing sample solutions and their distances.
    """
    problem_size = len(coords)
    samples = []
    
    for _ in range(num_samples):
        path = generate_random_solution(problem_size)
        distance = calculate_path_distance(coords, path)
        samples.append({
            "path": path,
            "distance": distance
        })
    
    # Sort samples by distance (ascending)
    samples.sort(key=lambda x: x["distance"])
    
    # Reverse to get descending order (as requested in the prompt format)
    samples.reverse()
    
    return samples

def format_coordinates(coords: Dict[str, List[float]]) -> str:
    """
    Format coordinates for the LLM prompt.
    
    Args:
        coords: Dictionary mapping node IDs to coordinate pairs.
        
    Returns:
        Formatted string of coordinates.
    """
    formatted_coords = []
    for node_id, (x, y) in coords.items():
        formatted_coords.append(f"({node_id}): ({int(x)}, {int(y)})")
    
    return ", ".join(formatted_coords)

def format_solution(path: List[int]) -> str:
    """
    Format a solution path for the LLM prompt.
    
    Args:
        path: List of node IDs representing a solution path.
        
    Returns:
        Formatted string of the solution path.
    """
    return ",".join(map(str, path))

def generate_llm_prompt(problem: Dict[str, Any], num_samples: int = 4) -> str:
    """
    Generate a prompt for an LLM to solve a TSP problem.
    
    Args:
        problem: Dictionary containing a TSP problem from the dataset.
        num_samples: Number of sample solutions to include in the prompt.
        
    Returns:
        Formatted prompt string for the LLM.
    """
    coords = {k: v for k, v in problem["coordinates"].items()}
    problem_size = len(coords)
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(coords)
    
    # Generate sample solutions
    samples = generate_sample_solutions(coords, num_samples)
    
    # Start building the prompt
    prompt = f"You are given a list of points with coordinates below: {format_coordinates(coords)}.\n\n"
    
    # Add distance matrix
    prompt += format_distance_matrix(distance_matrix, problem_size) + "\n"
    
    prompt += "Below are some previous traces and their lengths. The traces are arranged in descending order based on their lengths, where lower values are better.\n"
    
    # Add sample solutions
    for sample in samples:
        path_str = format_solution(sample["path"])
        distance = round(sample["distance"])
        prompt += f"{path_str} length: {distance}\n"
    
    # Add the request
    prompt += "\nGive me a new trace that is different from all traces above, and has a length lower than any of the above. "
    prompt += "The trace should traverse all nodes"
    prompt += "The path must start at 0 will return to 0 at the end of the trace, which is included in the distance."
    
    return prompt

def generate_llm_prompt_multi(problem: Dict[str, Any], num_samples: int = 3) -> str:
    """
    Generate a prompt for an LLM to solve a TSP problem.
    
    Args:
        problem: Dictionary containing a TSP problem from the dataset.
        num_samples: Number of sample solutions to include in the prompt.
        
    Returns:
        Formatted prompt string for the LLM.
    """
    coords = {k: v for k, v in problem["coordinates"].items()}
    problem_size = len(coords)
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(coords)
    
    # Generate sample solutions
    samples = generate_sample_solutions(coords, num_samples)
    
    # Start building the prompt
    SYSTEMPROMPT = f"""You are solving a Traveling Salesperson Problem (TSP). Your goal is to generate an optimized route with a lower total distance than any previously provided solution.\n\n
                   ## Problem Setup:\n
                   The problem consists of the following nodes with coordinates:\n{format_coordinates(coords)}.\n\n"""
    
    # Add distance matrix
    SYSTEMPROMPT += f"### Distance Matrix:\n {format_distance_matrix(distance_matrix, problem_size)} \n"


    SYSTEMPROMPT += """## Output Format (Strict):\n
                   You must respond with exactly one new valid route in this format:\n\n
                   **Route:** <trace> 0, X, Y, Z, ..., 0<trace>\n
                   ### Rules:\n
                   - The route must visit all nodes exactly once before returning to 0.\n
                   - The new route must have a total distance lower than the previously provided best distance.\n
                   - Do not repeat a previously suggested route.\n
                   - Only provide **one** improved route and no additional commentary.\n\n
                   Generate an improved route below:"""
    
    examples = {}
    # Add sample solutions
    for i, sample in enumerate(samples):
        path_str = format_solution(sample["path"])
        distance = round(sample["distance"])

        examples[path_str] = distance

    
    SYSTEMPROMPT += ''.join(
        f'\n\n <trace>{path},0<trace> \n length: \n {distance}' 
        for path, distance in examples.items()
    )


    USERPROMPT = '''Give me a new trace that is different from all traces above, and has a length lower than any of the
above. The trace should traverse all points exactly once. The trace should start with <trace> and end
with </trace>. Let's find a final solution step by step. Only put your final solution inside <trace> brackets.'''

    prompt  = [{"role": "system",
                "content": SYSTEMPROMPT},
                {'role': 'user',
                 'content': USERPROMPT}]
    
    # prompt += [
    #     item 
    #     for path, distance in examples.items() 
    #     for item in ({"role": "assisstant", "content": f'<trace>{path}<trace \n'}, {"role": "user", "content": f"Distance {distance} \n Find exactly one new route with a lower distance:"})
    # ]
    
    return prompt

def create_llm_dataset(dataset: Dict, output_filename: str = "tsp_llm_prompts.json", problems_per_size: int = 5) -> None:
    """
    Create a dataset of LLM prompts from the TSP dataset.
    
    Args:
        dataset: Dictionary containing the TSP dataset.
        output_filename: Path to save the LLM prompts dataset.
        problems_per_size: Number of problems to include for each size.
    """
    llm_dataset = {}
    
    sizes = [5, 10, 15, 20]
    pbar = tqdm(total=len(sizes) * problems_per_size, desc="Generating LLM Prompts")
    
    for size in sizes:
        size_key = f"size_{size}"
        llm_dataset[size_key] = []
        
        # Get problems of this size
        problems = dataset[size_key]
        
        # Select a subset of problems
        selected_problems = random.sample(problems, min(problems_per_size, len(problems)))
        
        for problem in selected_problems:
            prompt = generate_llm_prompt_multi(problem)
            
            # Add to the dataset
            llm_dataset[size_key].append({
                "problem_id": problem["problem_id"],
                "prompt": prompt,
                "size": size,
                "optimal_solution": {
                    "path": problem["solution"]["path"],
                    "distance": problem["solution"]["distance"]
                }
            })
            
            pbar.update(1)
    
    pbar.close()
    
    # Add metadata
    llm_dataset["metadata"] = {
        "total_prompts": sum(len(llm_dataset[f"size_{size}"]) for size in sizes),
        "sizes": sizes,
        "problems_per_size": problems_per_size,
        "description": "TSP LLM prompts dataset with distance matrices"
    }
    
    # Save the dataset
    with open(output_filename, 'w') as f:
        json.dump(llm_dataset, f, indent=2)
    
    print(f"LLM prompts dataset saved to {output_filename}")

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load the TSP dataset
    try:
        dataset = load_tsp_dataset("tsp_dataset_100_problems.json")
        print("TSP dataset loaded successfully.")
        
        # Create LLM prompts dataset
        create_llm_dataset(dataset, "tsp_llm_prompts_with_matrix_think.json", problems_per_size=25)
        
        # Print a sample prompt
        sample_size = 5  # Using a smaller size for clearer display
        sample_problem = dataset[f"size_{sample_size}"][0]
        sample_prompt = generate_llm_prompt(sample_problem)
        
        print("\nSample LLM Prompt for a size-5 problem:")
        print("-" * 80)
        print(sample_prompt)
        print("-" * 80)
        
    except FileNotFoundError:
        print("TSP dataset file not found. Please run the dataset generation script first.")
import json
import random
import time
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm
from tsp import calculate_tsp_distance, generate_tsp_distance_matrix

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

def load_tsp_prompt_dataset(filename: str = "tsp_llm_prompts.json") -> Dict:
    with open(filename, 'r') as f:
        dataset = json.load(f)
    return dataset

def format_distance_matrix(matrix: List[List[float]]) -> str:
    """
    Format the distance matrix with headers and column alignment.

    Args:
        matrix: The distance matrix.
    
    Returns:
        Formatted distance matrix with headers nad column alignment.
    """
    n = len(matrix)
    
    # Create default headers if none provided
    headers = [str(i) for i in range(n)]
        
    # Determine column width based on all values and headers
    max_width = max(
        max(len(f"{value:.1f}") for row in matrix for value in row),
        max(len(h) for h in headers)
    )
    
    # Create header row
    header_row = " " * (max_width + 1) + " ".join(h.rjust(max_width) for h in headers)
    
    # Build the matrix string with headers
    lines = [header_row]
    
    for i, row in enumerate(matrix):
        formatted_values = " ".join(f"{value:.1f}".rjust(max_width) for value in row)
        lines.append(f"{headers[i].ljust(max_width)} {formatted_values}")
    
    return "\n".join(lines)

def generate_random_solution(tsp: Dict[int, Tuple[int, int]]) -> List[int]:
    """
    Generate a random solution for a TSP problem, ensuring that all solutions start with and end with node 0.
    
    Args:
        tsp: Dictionary of node indices to coordinates
        
    Returns:
        List of node indices representing a random path through all nodes, starting with and ending with node 0.
    """
    if len(tsp) <= 2:
        raise Exception("Invalid tsp problem: please provide at least 3 nodes")
    
    nodes = list(range(1, len(tsp)))  # Exclude 0 from shuffling
    random.shuffle(nodes)
    return [0] + nodes + [0]

def generate_sample_solutions(tsp: Dict[int, Tuple[int, int]], num_samples: int = 4) -> List[Dict[str, Any]]:
    """
    Generate unique sample solutions for a TSP problem.
    
    Args:
        tsp: Dictionary of node indices to coordinates
        num_samples: Number of unique sample solutions to generate.
        
    Returns:
        List of dictionaries containing sample solutions and their distances.
    """
    samples = []
    
    for i in range(num_samples):
        path = generate_random_solution(tsp)
        for sample in samples:
            if sample['path'] == path:
                i -= 1
                continue
        distance = calculate_tsp_distance(tsp, path)
        samples.append({
            "path": path,
            "distance": distance
        })
    
    # Sort samples by distance (descending)
    samples.sort(key=lambda x: x["distance"], reverse=True)
    
    return samples

def format_coordinates(tsp: Dict[int, Tuple[int, int]]) -> str:
    """
    Format coordinates for the LLM prompt.
    
    Args:
        tsp: Dictionary of node indices to coordinates
        
    Returns:
        Formatted string of coordinates.
    """
    formatted_coords = []
    for node, (x, y) in tsp.items():
        formatted_coords.append(f"Node: {node}: ({x}, {y})")
    
    return "\n".join(formatted_coords)

def format_solution(path: List[int]) -> str:
    """
    Format a solution path for the LLM prompt.
    
    Args:
        path: List of nodes representing a solution path.
        
    Returns:
        Formatted string of the solution path.
    """
    return ",".join(map(str, path))

def generate_llm_prompt_old(tsp: Dict[int, Tuple[int, int]], num_samples: int = 4) -> str:
    """
    Generate a prompt for an LLM to solve a TSP problem.
    
    Args:
        tsp: Dictionary of node indices to coordinates
        num_samples: Number of sample solutions to include in the prompt.
        
    Returns:
        Formatted prompt string for the LLM.
    """
    # Create distance matrix
    distance_matrix = generate_tsp_distance_matrix(tsp)
    
    # Generate sample solutions
    samples = generate_sample_solutions(tsp, num_samples)
    
    # Start building the prompt
    prompt = f"## Nodes\nYou are given a list of points with coordinates below:\n{format_coordinates(tsp)}.\n\n"
    
    # Add distance matrix
    prompt += "## Distance Matrix\n" + format_distance_matrix(distance_matrix) + "\n\n"
    
    prompt += "Below are some previous traces and their lengths. The traces are arranged in descending order based on their lengths, where lower values are better.\n"
    
    # Add sample solutions
    for sample in samples:
        path_str = format_solution(sample["path"])
        distance = round(sample["distance"])
        prompt += f"{path_str} length: {distance}\n"
    
    # Add the request
    prompt += "\nGive me a new trace that is different from all traces above, and has a length lower than any of the above.\n"
    prompt += "The trace should traverse all nodes.\n"
    prompt += "The path must start at 0 will return to 0 at the end of the trace, which is included in the distance."
    
    return prompt

def generate_llm_prompt_new(tsp: Dict[int, Tuple[int, int]], num_samples: int = 3) -> str:
    """
    Generate a prompt for an LLM to solve a TSP problem.
    
    Args:
        tsp: Dictionary of node indices to coordinates
        num_samples: Number of sample solutions to include in the prompt.
        
    Returns:
        Formatted prompt string for the LLM.
    """
    # Create distance matrix
    distance_matrix = generate_tsp_distance_matrix(tsp)
    
    # Generate sample solutions
    samples = generate_sample_solutions(tsp, num_samples)
    
    # Start building the prompt
    SYSTEMPROMPT = f"""You are solving a Traveling Salesperson Problem (TSP). Your goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.

## Problem Setup
Given {len(tsp)} nodes with coordinates:\n{format_coordinates(tsp)}\n\n"""
    
    # Add distance matrix
    SYSTEMPROMPT += f"""
## Distance Matrix
The matrix below shows the distance between each pair of nodes:\n {format_distance_matrix(distance_matrix)} \n\n"""
    examples = {}

    # Add sample solutions
    for i, sample in enumerate(samples):
        path_str = format_solution(sample["path"])
        distance = round(sample["distance"])

        examples[path_str] = distance

    SYSTEMPROMPT += """
## Previous Solutions
These routes have already been tried:"""
    
    SYSTEMPROMPT += ''.join(
        f'\n\nRoute: {path} with total length: {distance}' 
        for path, distance in examples.items()
    )

    USERPROMPT = '''## Your Task
1. Analyze the distance matrix carefully
2. Apply systematic optimization techniques such as:
   - 2-opt or 3-opt local search
   - Nearest neighbor with look-ahead
   - Edge exchange optimization
   - Consider node clustering patterns
3. Calculate the total distance of your proposed route precisely

## Requirements
- Start and end at node 0
- Visit each node exactly once before returning to node 0
- Provide a route with a total distance lower than 1127
- Your solution must be different from the previous routes

## Output Format
Think through your approach step by step, showing your calculations. Then provide your final solution in exactly this format:

<trace>0,X,X,X,X,X,X,X,X,X,0</trace>'''

    prompt  = [
        {
            "role": "system",
            "content": SYSTEMPROMPT
        },
        {
            'role': 'user',
            'content': USERPROMPT
        }
    ]
    
    return prompt

def create_prompt_dataset(tsp_dataset: Dict, output_filename: str = "tsp_llm_prompts.json", problems_per_size: int = 5) -> None:
    """
    Create a dataset of LLM prompts from the TSP dataset.
    
    Args:
        dataset: Dictionary containing the TSP dataset.
        output_filename: Path to save the LLM prompts dataset.
        problems_per_size: Number of problems to include for each size.
    """
    prompt_dataset = []
    
    sizes = [5, 10, 15]
    pbar = tqdm(total=len(sizes) * problems_per_size, desc="Generating LLM Prompts")
    
    for size in sizes:
        size_key = f"size_{size}"
        
        # Get problems of this size
        problems = tsp_dataset[size_key]
        
        # Select a subset of problems
        selected_problems = random.sample(problems, min(problems_per_size, len(problems)))

        # Sort selected problems by problem id
        selected_problems = sorted(selected_problems, key=lambda x: int(x['problem_id'].split("_")[2]))
        
        # Generate llm prompt
        for problem in (selected_problems):
            prompt = generate_llm_prompt_new(problem['tsp'])
            
            # Add to the dataset
            prompt_dataset.append({
                "problem_id": problem["problem_id"],
                "size": size,
                "prompt": prompt,
                "solution": {
                    "path": problem["solution"]["path"],
                    "distance": problem["solution"]["distance"]
                }
            })
            
            pbar.update(1)
    
    pbar.close()
    
    
    # Save the dataset
    with open(output_filename, 'w') as f:
        json.dump(prompt_dataset, f, indent=2)
    
    print(f"LLM prompt dataset saved to {output_filename}")

def process_dataset(dataset: Dict) -> Dict:
    """
    Convert coordinates entry to tsp (Dict[int, Tuple[int, int]]) for every problem in dataset.
    Remove metadata entry from dataset.

    Args:
        dataset: Dictionary containing the TSP dataset.
    
    Returns:
        Processed dataset.
    """
    # Remove metadata entry
    del dataset['metadata']
    for size_key in dataset:
        problems = dataset[size_key]
        for problem in problems:
            # Replace coordinates with tsp
            coordinates = problem['coordinates']
            tsp = {}
            for node in coordinates:
                [x, y] = coordinates[node]
                tsp[int(node)] = (x, y)
            del problem['coordinates']
            problem['tsp'] = tsp

    return dataset

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Load the TSP dataset
        dataset = load_tsp_dataset("tsp_dataset_9000_problems.json")
        print("TSP dataset loaded successfully.")

        # Process dataset for compatibility with rest of script
        dataset = process_dataset(dataset)
        
        # Create LLM prompts dataset
        create_prompt_dataset(dataset, "tsp_llm_prompts_9000.json", problems_per_size=3000)
        
        # Print a sample prompt
        sample_size = 5  # Using a smaller size for clearer display
        sample_problem = dataset[f"size_{sample_size}"][0]
        sample_prompt = generate_llm_prompt_new(sample_problem['tsp'])
        
        print("\nSample LLM Prompt for a size-5 problem:")
        print("-" * 80)
        print(sample_prompt)
        print("-" * 80)
        
    except FileNotFoundError:
        print("TSP dataset file not found. Please run the dataset generation script first.")
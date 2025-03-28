import math
import re
from typing import Dict, List, Tuple
import numpy as np

def calculate_path_distance(coords: Dict[str, List[float]], path: List[int]) -> float:
    """
    Calculate the total distance of a path through the given coordinates.
    
    Args:
        coords: Dictionary mapping node IDs to coordinate pairs.
        path: List of node IDs representing the order of visitation.
        
    Returns:
        Total Euclidean distance of the path.
    """
    if not (isinstance(path, list)):
        return np.inf
    
    if not set(path) == set(range(len(path))):
        return np.inf
    
    total_distance = 0
    for i in range(len(path)):
        current_node = path[i]
        next_node = path[0] if i == len(path) - 1 else path[i + 1]
        
        # Fix the coordinate lookup to use the actual node IDs from the path
        x1, y1 = coords[str(current_node)]
        x2, y2 = coords[str(next_node)]
        
        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance
    
    return total_distance


# def generate_solutions(prompt_dict, model="llama3", max_new_tokens=300, temperature=0.5, n_samples=20):
#     """
#     Generates responses using the Ollama API with a given model.
#     """
#     # Convert structured messages to a properly formatted chat prompt
#     # formatted_prompt = "".join([msg["content"] for msg in prompt_dict])
    
#     if type(prompt_dict) == str:

#         prompt_dict = [{'role':'user', 'content': prompt_dict}]
#     solutions = []
#     for _ in range(n_samples):
#         response = ollama.chat(
#             model=model,
#             messages=prompt_dict,
#             options={"temperature": temperature, "num_predict": max_new_tokens}
#         )
#         solutions.append(response["message"]["content"])
    
#     return solutions

       
def extract_answer(text):
    """
    Extracts the last occurrence of a list inside <trace> brackets (both <trace>list<trace> and <trace>list</trace>)
    and removes trailing 0 if present.
    """
    matches = re.findall(r"<trace>([\d,]+)<\/trace>|<trace>([\d,]+)<trace>", text)
    if not matches:
        return []
    
    # Extract the last non-empty match
    last_match = [m for match in matches for m in match if m][-1]
    numbers = last_match.split(",")
    
    # Remove trailing '0' if present
    if numbers and numbers[-1] == "0":
        numbers.pop()
    
    return [int(num) for num in numbers]

def extract_total_length(text):
    return [int(match) for match in re.findall(r'total length:\s*(\d+)', text)]


def extract_nodes(text_string):
    # Find the line that starts with "Given"
    lines = text_string.split('\n')
    node_lines = []
    
    # Flag to track when we're in the nodes section
    capturing = False
    
    # Find and collect the node lines
    for line in lines:
        if line.startswith("Given"):
            capturing = True
            continue
        
        if capturing and line.startswith("Node:"):
            node_lines.append(line)
        
        # Stop capturing when we hit an empty line after finding nodes
        if capturing and line.strip() == "":
            break
    
    # If we didn't find any nodes through the first approach, try direct pattern matching
    if not node_lines:
        import re
        node_pattern = r"Node: (\d+): \((-?\d+), (-?\d+)\)"
        node_lines = re.findall(node_pattern, text_string)
        
        # Convert regexp results to dictionary directly if we found matches
        if node_lines:
            return {int(node): (int(x), int(y)) for node, x, y in node_lines}
    
    # Process the node lines
    nodes_dict = {}
    for line in node_lines:
        # Extract node number and coordinates
        parts = line.split(": ")
        node_num = int(parts[1].split(":")[0])
        
        # Extract coordinates - handling the parentheses
        coord_str = parts[2].strip()
        coord_str = coord_str.strip("()")
        x, y = map(int, coord_str.split(", "))
        
        nodes_dict[node_num] = (x, y)
    
    return nodes_dict

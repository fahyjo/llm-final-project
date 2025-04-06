from unsloth import FastLanguageModel
import torch
from tqdm import tqdm
from typing import List, Dict, Any
import json
import re
from grpo.grpo import SYSTEM_PROMPT, extract_trace, optimal_solution_reward_func, improvement_reward_func, valid_response_reward_func, strict_format_reward_func, soft_format_reward_func
from tsp.tsp_llm import coordinates_to_tsp
from tsp.tsp import calculate_tsp_distance


MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
BENCHMARK_DATASET = "tsp_benchmark_dataset.json"
BENCHMARK_RESULTS = "tsp_benchmark_results_qwen2.5-7B-2.json"


def load_benchmark_dataset(filename: str) -> Dict:
    """
    Load the benchmark dataset.
    
    Args:
        filename: Path to the benchmark dataset
        
    Returns:
        Dict: Benchmark dataset
    """
    with open(filename, 'r') as f:
        dataset = json.load(f)
    return dataset

def save_benchmark_results(filename: str, results: Dict) -> None:
    """
    Save the benchmark results.

    Args:
        filename: Path to save benchmark results to
        results: The benchmark results
    """

    with open(filename, "w") as file:
        json.dump(results, file, indent=4)

def extract_assistant_response_regex(text: str) -> str:
    """
    Extract the assistant's response from the decoded text output of Qwen.
    
    Args:
        text: The decoded text output from the model
        
    Returns:
        str: The extracted assistant response or empty string if no match found
    """
    # Common patterns for assistant responses in chat models
    patterns = [
        r"(?:^|\n)(?:assistant|assistant:)(.*?)(?:$|\n\s*(?:user|user:|system|system:|<\|im_end\|>))",  # Matches cases with various endings
        r"<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|$)",  # Matches with special tokens
        r"(?:^|\n)assistant:\s*(.*?)(?:$|\n\s*(?:user|user:|system|system:))",  # Standard chat format
    ]
    
    # Try each pattern until we find a match
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches.group(1).strip()
    
    # If no structured format is found, return everything after the last "user:" or system prompt
    last_user = re.search(r"(?:^|\n)user:(?!.*\nuser:).*?\n(.*?)$", text, re.DOTALL | re.IGNORECASE)
    if last_user:
        return last_user.group(1).strip()
    
    # If all else fails, return the original text (this is a fallback)
    return text.strip()

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 3500, num_samples: int = 3, temperature: float = 0.7, top_p: float = 0.9) -> Dict[str, Any]:
    """
    Generate a response from the model based on the prompt.

    Args:
        model: The model
        tokenizer: The tokenizer
        prompt: The input prompt
        max_new_tokens: Max number of tokens to generate
        num_samples: Number of samples per prompt
        temperature: Model temperature
        top_p: Model top_p
    
    Returns:
        Dict[str, Any]: Summary dict containing input token count, average output token count, and
                        model responses
    """

    # Format chat message for Qwen
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    # Tokenize input and move input tensors to GPU
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # Get input token count
    input_token_count = inputs.shape[1]

    # Generate outputs
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True
    )
        
    # Get output token counts
    output_sequences = outputs.sequences
    avg_output_token_count = sum([len(o) - input_token_count for o in output_sequences]) / len(output_sequences)
        
    # Process outputs
    responses = [tokenizer.decode(o, skip_special_tokens=True) for o in output_sequences]
    responses = [extract_assistant_response_regex(r) for r in responses]

    # Return token counts and responses
    out = {
        "input token count": input_token_count,
        "average output token count": avg_output_token_count,
        "responses": responses
    }

    return out

def apply_reward_functions(problem: Dict, responses: List[str]) -> Dict:
    """
    Evaluate model responses with GRPO reward functions.

    Args:
        problem: The problem
        responses: The model responses
    
    Returns:
        Dict: Summary dict containing rewards averaged across responses and rewards per response
    """

    # Extract problem info
    n = len(responses)
    tsp = coordinates_to_tsp(problem['coordinates'])

    # Extract traces and distances from model responses
    traces = [extract_trace(response) for response in responses]
    distances = [round(calculate_tsp_distance(tsp, trace)) for trace in traces]
    
    # Construct args for reward functions
    completions = [[{'content': response}] for response in responses]
    kwargs = {
        'tsp': [tsp for _ in responses],
        'answer': [problem['solution'] for _ in responses],
        'reference_distance': [problem['reference_distance'] for _ in responses]
    }

    # Invoke reward functions
    optimal_solution_rewards = optimal_solution_reward_func(completions, **kwargs)
    improvement_rewards = improvement_reward_func(completions, **kwargs)
    valid_response_rewards = valid_response_reward_func(completions, **kwargs)
    strict_format_rewards = strict_format_reward_func(completions, **kwargs)
    soft_format_rewards = soft_format_reward_func(completions, **kwargs)

    # Calculate problem averages
    problem_summary = {
        "problem_id": problem["problem_id"],
        "solution": problem["solution"],
        "average optimal solution reward": sum(optimal_solution_rewards) / n,
        "average improvement reward": sum(improvement_rewards) / n,
        "average valid response reward": sum(valid_response_rewards) / n,
        "average strict format reward": sum(strict_format_rewards) / n,
        "average soft format reward": sum(soft_format_rewards) / n
    }

    # Add summary for each sample
    for i in range(n):
        sample_key = f"sample_{i}"
        problem_summary[sample_key] = {
            "response": responses[i],
            "solution": {
                "path": traces[i],
                "distance": distances[i]
            },
            "optimal solution reward": optimal_solution_rewards[i],
            "improvement reward": improvement_rewards[i],
            "valid response reward": valid_response_rewards[i],
            "strict format reward": strict_format_rewards[i],
            "soft format reward": soft_format_rewards[i],
        }
    
    return problem_summary

def solve_benchmark(model, tokenizer, dataset: Dict) -> Dict:
    """
    Solve the benchmark.

    Args:
        model: The model
        tokenizer: The tokenizer
      
    Return:
        Dict: Summary dict containing benchmark results per problem size
    """

    # Set up progress bar
    pbar = tqdm(total=len(dataset) * len(dataset["size_5"]), desc="Solving TSP Benchmark")

    # Summary dict
    results = {}

    # Solve benchmark dataset
    for size_key in dataset:
        problems = dataset[size_key]

        size_results = []
        for problem in problems:
            prompt = problem['prompt']

            # Generate 3 completions per prompt
            generate_out = generate(model, tokenizer, prompt)

            # Calculate rewards for model responses
            reward_out = apply_reward_functions(problem, generate_out['responses'])

            # Remove completions from gen out
            del generate_out['responses']

            # Append problem summary to size results
            generate_out.update(reward_out)
            problem_summary = generate_out
            size_results.append(problem_summary)

            # Update progress bar
            pbar.update(1)
      
        results[size_key] = size_results
      
    # Close progress bar
    pbar.close()

    # Return results
    return results

def summarize_results(results: Dict) -> Dict:
    """
    Summarize the benchmark results for each problem size.

    Args:
        results: The benchmark results

    Returns:
        Dict: Summary dict containing averaged rewards for each problem size
    """

    # Summary
    summary = {}

    for size_key in results:
        size_results = results[size_key]
        size_summary = {}
        n = len(size_results)

        # Calculate summary per size
        average_input_token_count = sum([problem['input token count'] for problem in size_results]) / n
        average_output_token_count = sum([problem['average output token count'] for problem in size_results]) / n
        average_optimal_solution_reward = sum([problem['average optimal solution reward'] for problem in size_results]) / n
        average_improved_reward = sum([problem['average improvement reward'] for problem in size_results]) / n
        average_valid_response_reward = sum([problem['average valid response reward'] for problem in size_results]) / n
        average_strict_format_reward = sum([problem['average strict format reward'] for problem in size_results]) / n
        average_soft_format_reward = sum([problem['average soft format reward'] for problem in size_results]) / n

        # Load size summary
        size_summary['average input token count'] = average_input_token_count
        size_summary['average output token count'] = average_output_token_count
        size_summary['average optimal solution reward'] = average_optimal_solution_reward
        size_summary['average improvement reward'] = average_improved_reward
        size_summary['average valid response reward'] = average_valid_response_reward
        size_summary['average strict format reward'] = average_strict_format_reward
        size_summary['average soft format reward'] = average_soft_format_reward

        # Append size summary to summary
        summary[size_key] = size_summary

    # Append summary to results
    results['summary'] = summary

    return results

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=4096,
        load_in_4bit=True,
        dtype=torch.bfloat16
    )

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # Load benchmark dataset
    dataset = load_benchmark_dataset(BENCHMARK_DATASET)

    # Solve benchmark
    results = solve_benchmark(model, tokenizer, dataset)

    # Summarize results
    results = summarize_results(results)

    # Save benchmark results
    save_benchmark_results(BENCHMARK_RESULTS, results)
from unsloth import FastLanguageModel
import torch
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import wandb
import re
import json
import random
import numpy as np
from typing import Dict, List
from tsp.tsp import calculate_tsp_distance
from tsp.tsp_llm import coordinates_to_tsp


MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
TRAINING_DATASET = "grpo/datasets/tsp_training_dataset.json"
TRAINING_OUTPUT_DIR = "grpo/out/Qwen2.5-7B-Instruct-GRPO"

WANDB_PROJECT_NAME = "Qwen2.5-7B-Instruct-GRPO"
WANDB_RUN_NAME = "Size-5-Baseline"

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the assistant solves it.
 The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
 The reasoning process and answer are enclosed within <reasoning> </reasoning> and <trace> </trace> tags, respectively, i.e.,
 <reasoning>
 reasoning process here
 </reasoning>
 <trace>
 answer here
 </trace>"""
STRICT_FORMAT_REGEX = r"^<reasoning>\n.*?\n</reasoning>\n<trace>\n.*?\n</trace>$"
SOFT_FORMAT_REGEX = r"<reasoning>.*?</reasoning>\s*<trace>.*?</trace>"


def load_training_dataset(filename: str) -> Dict:
    """
    Load the training dataset.

    Args: 
        filename: File containing training dataset

    Returns:
        Dict: Training dataset
    """

    # Open training prompt dataset
    with open(filename, 'r') as f:
        dataset = f.read()
        dataset = json.loads(dataset)

    # Load all problems into one list
    problems = [p for size_key in dataset for p in dataset[size_key]]

    # Shuffle problems
    random.shuffle(problems)
    
    # Format prompts
    training_dataset = [{
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': p['prompt']}
        ],
        'answer': p['solution'],
        'tsp': coordinates_to_tsp(p['coordinates']),
        'reference_distance': p['reference_distance']
    } for p in problems]
    
    return training_dataset 

def extract_trace(response: str) -> List[int]:
    """
    Extracts the last occurrence of a list of ints inside <trace></trace> or <trace><trace> brackets.

    Args:
        response: The model response
    
    Returns:
        List[int]: The last trace provided by the model
    """

    # Regex pattern to match <trace>...</trace> and <trace>...<trace>
    matches = re.findall(r"<trace>\s*([\d]+(?:\s*,\s*[\d]+)*)\s*</trace>|<trace>\s*([\d]+(?:\s*,\s*[\d]+)*)\s*<trace>", response)
    if not matches:
        return []
    
    # Extract the last non-empty match
    last_match = next(filter(None, matches[-1]))

    # Convert to list of integers
    return [int(num) for num in re.split(r"\s*,\s*", last_match)]

def optimal_solution_reward_func(completions, **kwargs) -> list[float]:
    """ Score = 2.5 if response solution is within 10 percent of optimal solution, 0.0 otherwise """

    # Get responses
    responses = [completion[0]['content'] for completion in completions]

    # Get tsp, optimal distance, target distance from extra parms
    tsp = kwargs['tsp'][0]
    optimal_distance = kwargs['answer'][0]['distance']
    target_distance = optimal_distance * 1.1

    # Extract trace from each response and calculate its distance
    traces = [extract_trace(response) for response in responses]

    # Determine if each response is a valid response
    valid_response_rewards = valid_response_reward_helper(traces, len(tsp) + 1)

    # Calculate distance for each valid trace
    distances = [round(calculate_tsp_distance(tsp, traces[i])) if valid_response_rewards[i] == 0.5 else np.inf for i in range(len(traces))]
    
    # 2.5 for each response that meets the target distance
    return [2.5 if distance <= target_distance else 0.0 for distance in distances]

def improvement_reward_func(completions, **kwargs) -> list[float]:
    """ Score = 2.5 if response solution improves on provided solutions, 0.0 otherwise """

    # Get responses
    responses = [completion[0]['content'] for completion in completions]

    # Get tsp and reference distance from extra params
    tsp = kwargs['tsp'][0]
    reference_distance = kwargs['reference_distance'][0]

    # Extract trace from each response
    traces = [extract_trace(response) for response in responses]

    # Determine if each response is a valid response
    valid_response_rewards = valid_response_reward_helper(traces, len(tsp) + 1)

    # Calculate distance for each valid trace
    distances = [round(calculate_tsp_distance(tsp, traces[i])) if valid_response_rewards[i] == 0.5 else np.inf for i in range(len(traces))]

    # 2.5 for each response that meets the reference distance
    return [2.5 if distance <= reference_distance else 0.0 for distance in distances]

def valid_response_reward_func(completions, **kwargs) -> list[float]:
    """ Score = 0.5 if response solution contains trace with correct length, starts with node 0,
    ends with node 0, and is a valid path, 0.0 otherwise """

    # Get responses
    responses = [completion[0]['content'] for completion in completions]

    # Get expected trace length from extra params
    trace_length = len(kwargs['tsp'][0]) + 1

    # Extract trace from each response
    traces = [extract_trace(response) for response in responses]
    
    # 0.5 if the trace has the correct length, starts with node 0, ends with node 0, and is a valid path
    return valid_response_reward_helper(traces, trace_length)

def valid_response_reward_helper(traces: List[List[int]], trace_length: int) -> List[float]:
    """ Score = 0.5 if response solution contains trace with correct length, starts with node 0,
    ends with node 0, and is a valid path, 0.0 otherwise """
    return [
        0.5 if len(trace) == trace_length
        and trace[0] == 0
        and trace[-1] == 0
        and set(trace) == set(range(trace_length - 1)) else 0.0 for trace in traces]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """ Score = 0.25 if response solution matches requested solution format, 0.0 otherwise """
    responses = [completion[0]["content"] for completion in completions]
    pattern = STRICT_FORMAT_REGEX
    matches = [re.match(pattern, response, flags=re.DOTALL) for response in responses] 
    return [0.25 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """ Score = 0.25 if response solution loosely matches request solution format, 0.0 otherwise """
    responses = [completion[0]["content"] for completion in completions]
    pattern = SOFT_FORMAT_REGEX
    matches = [re.match(pattern, response, flags=re.DOTALL) for response in responses] 
    return [0.25 if match else 0.0 for match in matches]


if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL,
        max_seq_length = 4200,        # Can increase for longer reasoning traces
        load_in_4bit = True,          # False for LoRA 16
        fast_inference = True,        # Enable vLLM fast inference
        max_lora_rank = 64,           # Larger rank = smarter, but slower
        gpu_memory_utilization = 0.5, # Reduce if out of memory
        dtype=torch.bfloat16
    )

    # Apply PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,                                     # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],                                          # Remove QKVO if out of memory
        lora_alpha = 64,
        use_gradient_checkpointing = "unsloth",     # Enable long context finetuning
        random_state = 3407,
    )

    # Load training dataset
    dataset = load_training_dataset(TRAINING_DATASET)

    # Reward functions
    reward_funcs = [
        optimal_solution_reward_func,
        improvement_reward_func,
        valid_response_reward_func,
        strict_format_reward_func,
        soft_format_reward_func
    ]

    # Load env variables
    load_dotenv()

    # Initialize wandb project
    wandb.init(
        project=WANDB_PROJECT_NAME,
        name=WANDB_RUN_NAME
    )

    # Load GRPO Config
    training_args = GRPOConfig(
        use_vllm = True,                 # Use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 4,             # Decrease if out of memory
        max_prompt_length = 600,         # SPECIFICALLY FOR SIZE 5
        max_completion_length = 3500,
        num_train_epochs = 1,            # Set to 1 for a full training run
        max_steps = len(dataset),
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "wandb",              # Can use Weights & Biases
        output_dir = TRAINING_OUTPUT_DIR,
        temperature=0.7,
        beta=0.0
    )

    # Load GRPO Trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = reward_funcs,
        args = training_args,
        train_dataset = dataset,
    )

    # Train model
    trainer.train()
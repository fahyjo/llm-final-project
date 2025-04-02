# train_grpo.py
#
# See https://github.com/willccbb/verifiers for ongoing developments
#
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import json
from utils import extract_trace, extract_total_length, extract_nodes
from tsp import calculate_tsp_distance
import random
import numpy as np

# Load and prep dataset

SYSTEM_PROMPT = """
    first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning
    process and answer are enclosed within <reasoning> </reasoning> and <trace> </trace> tags, respectively, i.e.,
    <reasoning> reasoning process here </reasoning><trace> answer here </trace>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<trace>
{answer}
</trace>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()



def get_TSP_questions(filename = "tsp_prompt_training_dataset.json"):
    with open(filename, 'r') as f:
        data = f.read()
        data = json.loads(data)

    prompts = []
    
    for val in data.values():
        prompts += val
        
    random.shuffle(prompts)
    
    prompts = [{ # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['prompt']}
        ],
        'answer': x['solution']
    } for x in prompts]
    
    return prompts 

def optimal_solution_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    coords = [extract_nodes(prompt[1]['content']) for prompt in prompts]
    valid_response_rewards = valid_response_reward_helper_func(prompts, responses)
    extracted_responses = [extract_trace(r) for r in responses]
    distances = [calculate_tsp_distance(c, p) if valid_response_rewards[i] == 1.0 else np.inf for i, (p, c) in enumerate(zip(extracted_responses, coords))]
    best_dists = [a['distance'] for a in answer]
    
    return [2.0 if ((valid_response_rewards[i] == 1.0) and (distances[i] < best_dists[i])) else 0.0 for i in range(len(responses))]

def improvement_reward_func(prompts, completions, answer, **kwargs) -> list[float]:

    responses = [completion[0]['content'] for completion in completions]
    coords = [extract_nodes(prompt[1]['content']) for prompt in prompts]
    valid_response_rewards = valid_response_reward_helper_func(prompts, responses)
    prev_best = [min(extract_total_length(prompt[1]['content'])) for prompt in prompts]
    extracted_responses = [extract_trace(r) for r in responses]
    distances = [calculate_tsp_distance(c, p) if valid_response_rewards[i] == 1.0 else np.inf for i, (p, c) in enumerate(zip(extracted_responses, coords))]
    return [2.0 if ((valid_response_rewards[i] == 1.0) and (distances[i] < prev_best[i])) else 0.0 for i in range(len(responses))]

def valid_response_reward_func(prompts, completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    num_coords = [len(extract_nodes(prompt[1]['content'])) for prompt in prompts]
    traces = [extract_trace(r) for r in responses]
    
    
    # Create the result list explicitly instead of in the return statement
    result = []
    for trace, num_coord in zip(traces, num_coords):
        if (len(trace) == num_coord+1 and 
            trace[0] == 0 and 
            trace[-1] == 0 and 
            set(trace) == set(range(num_coord))):
            result.append(1.0)
        else:
            result.append(0.0)
                
    return result

def valid_response_reward_helper_func(prompts, responses, **kwargs) -> list[float]:
    num_coords = [len(extract_nodes(prompt[1]['content'])) for prompt in prompts]
    traces = [extract_trace(r) for r in responses]
    
    # Create the result list explicitly instead of in the return statement
    result = []
    for trace, num_coord in zip(traces, num_coords):
        if (len(trace) == num_coord+1 and 
            trace[0] == 0 and 
            trace[-1] == 0 and 
            set(trace) == set(range(num_coord))):
            result.append(1.0)
        else:
            result.append(0.0)
    
    return result


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<trace>\n.*?\n</trace>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<trace>.*?</trace>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]


def main():
    #model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    dataset = get_TSP_questions()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if "Llama" in model_name:
        output_dir = "outputs/Llama-1B-GRPO"
        run_name = "Llama-1B-GRPO-gsm8k"
    else:
        output_dir="outputs/Qwen-1.5B-GRPO"
        run_name="Qwen-1.5B-GRPO-gsm8k"
        
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None
    ).to(device)
            
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # use peft at your own risk; not working for me with multi-GPU training
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            optimal_solution_reward,
            improvement_reward_func],
        args=training_args,
        train_dataset=dataset,
        #peft_config=peft_config
    )
    trainer.train()
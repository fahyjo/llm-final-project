from dataclasses import dataclass
from typing import List
from .grpo import optimal_solution_reward_func, improvement_reward_func, valid_response_reward_func, strict_format_reward_func, soft_format_reward_func

@dataclass
class GRPORunConfig:
    model: str
    training_output_dir: str
    wandb_project_name: str
    wandb_run_name: str
    max_seq_length: int
    max_lora_rank: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    num_train_epochs: int
    reward_funcs: List[function]
    reward_weights: List[float]
    lr_scheduler_type: str
    learning_rate: float

config1 = GRPORunConfig(
    model = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
    training_output_dir = "grpo/out/Qwen2.5-3B-Instruct/run2",
    wandb_project_name = "Qwen2.5-3B-Instruct",
    wandb_run_name = "run2",
    max_seq_length = 2600,
    max_lora_rank = 64,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 1,
    num_generations = 8,
    max_prompt_length = 600,
    max_completion_length = 2000,
    num_train_epochs = 2,
    reward_funcs = [
        optimal_solution_reward_func,
        improvement_reward_func,
        valid_response_reward_func,
        strict_format_reward_func,
        soft_format_reward_func
    ],
    reward_weights = [2.5, 2.5, 0.5, 0.5, 0.5],
    learning_rate=5e-6,
    lr_scheduler_type="cosine"
)

config2 = GRPORunConfig(
    model = "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
    training_output_dir = "grpo/out/Qwen2.5-3B-Instruct/runjacob1",
    wandb_project_name = "Qwen2.5-3B-Instruct",
    wandb_run_name = "runjacob1",
    max_seq_length = 2600,
    max_lora_rank = 64,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 1,
    num_generations = 8,
    max_prompt_length = 600,
    max_completion_length = 2000,
    num_train_epochs = 1,
    reward_funcs = [
        optimal_solution_reward_func,
        improvement_reward_func,
        valid_response_reward_func,
        strict_format_reward_func,
        soft_format_reward_func
    ],
    reward_weights = [2.5, 2.5, 0.35, 0.35, 0.35],
    learning_rate=1e-5,
    lr_scheduler_type="constant"
)
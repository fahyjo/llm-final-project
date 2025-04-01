import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from utils import calculate_path_distance, extract_answer, extract_total_length, extract_nodes


SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <reasoning> </reasoning> and <trace> </trace> tags, respectively, i.e.,
<reasoning>
reasoning process here
</reasoning>
<trace>
answer here
</trace>
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



def get_TSP_questions():
    with open('tsp_llm_prompts_9000.json', 'r') as f:
        data = f.read()
        data = json.loads(data)

    data = [{ # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['prompt'][0]['content']+x['prompt'][1]['content']}
        ],
        'answer': x['solution']
    } for x in data]
    
    return data # type: ignore


# reward functions
def optimal_solution_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    coords = extract_nodes(prompts[0][1]['content'])
    extracted_responses = [extract_answer(r) for r in responses]
    distances = [calculate_path_distance(coords, p) for p in extracted_responses]
    return [2.0 if d-.1 <= answer[0]['distance'] else 0.0 for d in distances]

def improvement_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    prev_best = min(extract_total_length(prompts[0][1]['content']))
    coords = extract_nodes(prompts[0][1]['content'])
    extracted_responses = [extract_answer(r) for r in responses]
    distances = [calculate_path_distance(coords, p) for p in extracted_responses]
    return [2.0 if d < prev_best else 0.0 for d in distances]

def valid_response_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    num_coords = len(extract_nodes(prompts[0][1]['content']))
    extracted_responses = [extract_answer(r) for r in responses]
    return [1.0 if len(r) == num_coords and set(r) == set(range(num_coords)) else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<trace>\n.*?\n</trace>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<trace>.*?</trace>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<trace>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</trace>\n")[-1])*0.001
    if text.count("\n</trace>") == 1:
        count += 0.125
        count -= (len(text.split("\n</trace>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
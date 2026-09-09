from es import * 
from rollout import * 
from env import *
from agent import *

def env_fn(): 
    solution = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ]

    puzzle = [
        [1, 0, 3, 4],
        [3, 4, 1, 0],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ]

    return SudokuEnv(
        puzzle=puzzle,
        solution=solution,
        max_steps=6,
    )
    
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

model.eval()
    
    
    
agent = QwenSudokuAgent(
    model=model,
    tokenizer=tokenizer,
)

es = ESOptimizer(
    model=model,
    sigma=1e-3,
    alpha=5e-4,
    seed=42,
)


seeds, rewards = evaluate_population(
    env_fn=env_fn,
    agent=agent,
    es_optimizer=es,
    population_size=2,
)

print("Seeds:", seeds)
print("Rewards:", rewards)
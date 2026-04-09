import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# Environment variables as per Mandatory Instructions
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Environment settings
TASK_NAME = "cpu-scheduling"
BENCHMARK = "meta-hackathon-v1"
MAX_STEPS = 50
TEMPERATURE = 0.2

# Success threshold (normalized score)
SUCCESS_SCORE_THRESHOLD = 0.3 
MAX_TOTAL_REWARD = 500.0  # Estimated max reward for normalization

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert CPU Task Scheduler. Your goal is to choose the best task to run from the ready queue.
    
    Objectives:
    1. Minimize average waiting time.
    2. Prioritize high-importance (high priority) tasks.
    3. Ensure fairness and avoid starvation (don't ignore tasks for too long).
    
    Observation Format:
    The observation is a list of tasks. Each task has: [remaining_time, waiting_time, priority].
    Indices correspond to the index in the task queue.
    
    Response Format:
    You MUST respond with a JSON object containing:
    - "task_index": The integer index of the task you want to schedule.
    - "reason": A short human-readable explanation for your choice.
    
    Example: {"task_index": 2, "reason": "Task 2 has the highest priority and has been waiting for 10 units."}
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def parse_observation(obs_data: List[float]) -> str:
    tasks = []
    # Observation is [rem, wait, prio] * 10
    for i in range(0, len(obs_data), 3):
        rem, wait, prio = obs_data[i:i+3]
        if rem > 0:
            tasks.append(f"Task {i//3}: Remaining={rem}, Waiting={wait}, Priority={prio}")
    return "\n".join(tasks) if tasks else "No tasks in queue."

def get_model_decision(client: OpenAI, obs_text: str):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current Task Queue:\n{obs_text}\n\nDecision:"},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        return int(data.get("task_index", 0)), data.get("reason", "No reason provided.")
    except Exception as e:
        # Fallback to index 0 if LLM fails
        return 0, f"Error in LLM decision: {e}"

async def main():
    # In a real OpenEnv submission, we interact with the server.
    # For local validation, we can use the environment directly.
    from env import CPUSchedulerEnv
    from models import Action
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CPUSchedulerEnv()
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    obs = env.reset()
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        for step in range(1, MAX_STEPS + 1):
            obs_text = parse_observation(obs.data)
            task_idx, reason = get_model_decision(client, obs_text)
            
            # Step the environment
            next_obs, reward, done, info = env.step(task_idx)
            
            # The environment now provides its own explanation in info
            # We combine it with the agent's reason
            full_action_desc = f"Selected {task_idx} - {reason} | Env: {info.get('explanation', '')}"
            
            log_step(step=step, action=full_action_desc, reward=reward, done=done, error=None)
            
            rewards.append(reward)
            steps_taken = step
            obs = next_obs
            
            if done:
                break
                
        total_reward = sum(rewards)
        score = min(max(total_reward / MAX_TOTAL_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())

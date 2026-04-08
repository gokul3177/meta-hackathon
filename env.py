import numpy as np
import random
from fastapi import FastAPI
from models import Task, EnvState, Action, Observation, StepResponse, ResetResponse

app = FastAPI(title="CPU Scheduler RL Environment")

class CPUSchedulerEnv:
    def __init__(self, max_tasks=10):
        self.max_tasks = max_tasks
        self.reset()

    def reset(self):
        self.current_time = 0
        self.task_queue = []
        self.finished_tasks = []
        
        # Generate 3-5 random tasks
        num_tasks = random.randint(3, 5)
        for i in range(num_tasks):
            burst = random.randint(2, 10)
            task = Task(
                task_id=i,
                arrival_time=0,
                burst_time=burst,
                remaining_time=burst,
                waiting_time=0
            )
            self.task_queue.append(task)
            
        return self._get_observation()

    def _get_observation(self):
        # Observation is a flat list of [remaining_time, waiting_time] for each task slot
        obs_data = []
        for i in range(self.max_tasks):
            if i < len(self.task_queue):
                task = self.task_queue[i]
                obs_data.extend([float(task.remaining_time), float(task.waiting_time)])
            else:
                obs_data.extend([0.0, 0.0]) # Padding
        return Observation(data=obs_data)

    def step(self, action_idx):
        reward = 0
        done = False
        
        # If queue empty but env not done (shouldn't happen with proper logic)
        if not self.task_queue:
            return self._get_observation(), -5.0, True, {"msg": "No tasks left"}

        # Action corresponds to task index in the current queue
        # If action is invalid (out of bounds), penalize and pick the first one or stay idle
        if action_idx < 0 or action_idx >= len(self.task_queue):
            reward -= 2.0
            actual_action = 0 # Fallback
        else:
            actual_action = action_idx

        # Execute selected task for 1 unit
        task = self.task_queue[actual_action]
        task.remaining_time -= 1
        self.current_time += 1
        
        # Update waiting times for OTHER tasks in queue
        for i, t in enumerate(self.task_queue):
            if i != actual_action:
                t.waiting_time += 1
        
        # Reward logic
        reward -= 1.0 # Constant penalty for time passing
        
        # Check if task completed
        if task.remaining_time <= 0:
            reward += 10.0
            self.finished_tasks.append(task)
            self.task_queue.pop(actual_action)
            
        # Check if all tasks finished
        if not self.task_queue:
            done = True
            
        return self._get_observation(), float(reward), done, {}

# Initialize global environment instance
env = CPUSchedulerEnv()

@app.post("/reset", response_model=ResetResponse)
def reset():
    obs = env.reset()
    return ResetResponse(observation=obs)

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    obs, reward, done, info = env.step(action.task_index)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

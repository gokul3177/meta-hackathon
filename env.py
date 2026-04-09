import numpy as np
import random
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openenv.env.env import Env
from models import Task, EnvState, Action, Observation, StepResponse, ResetResponse, StateResponse, TaskStatus

app = FastAPI(title="Real-World CPU Scheduler RL Environment")

class CPUSchedulerEnv(Env):
    def __init__(self, max_tasks=10):
        self.max_tasks = max_tasks
        self.total_reward = 0.0
        self.reset()

    def reset(self):
        self.current_time = 0
        self.task_queue = []
        self.finished_tasks = []
        self.total_reward = 0.0
        
        # Generate 3-5 initial random tasks
        num_tasks = random.randint(3, 5)
        for i in range(num_tasks):
            self._generate_task(i)
            
        return self._get_observation()

    def _generate_task(self, task_id_prefix=None):
        if len(self.task_queue) >= self.max_tasks:
            return None
            
        tid = task_id_prefix if task_id_prefix is not None else random.randint(100, 999)
        burst = random.randint(2, 10)
        priority = random.randint(1, 10)
        
        task = Task(
            task_id=tid,
            arrival_time=self.current_time,
            burst_time=burst,
            remaining_time=burst,
            waiting_time=0,
            priority=priority,
            status=TaskStatus.READY
        )
        self.task_queue.append(task)
        return task

    def state(self):
        # Calculate system metrics
        total_wait = sum(t.waiting_time for t in self.finished_tasks + self.task_queue)
        total_tasks = len(self.finished_tasks) + len(self.task_queue)
        avg_wait = total_wait / total_tasks if total_tasks > 0 else 0
        
        # Starvation risk: percentage of tasks currently exceeding their dynamic threshold
        starving_count = sum(1 for t in self.task_queue if t.waiting_time > max(15, 2 * t.burst_time))
        starvation_risk = starving_count / len(self.task_queue) if self.task_queue else 0.0
        
        cpu_util = 1.0 if len(self.task_queue) > 0 else 0.0
        
        return EnvState(
            current_time=self.current_time,
            task_queue=self.task_queue,
            finished_tasks=self.finished_tasks,
            cpu_utilization=cpu_util,
            avg_waiting_time=avg_wait,
            starvation_risk=starvation_risk,
            total_reward=self.total_reward
        )

    def _get_observation(self):
        # [remaining_time, waiting_time, priority] per task slot
        obs_data = []
        for i in range(self.max_tasks):
            if i < len(self.task_queue):
                task = self.task_queue[i]
                obs_data.extend([
                    float(task.remaining_time), 
                    float(task.waiting_time),
                    float(task.priority)
                ])
            else:
                obs_data.extend([0.0, 0.0, 0.0]) # Padding
        return Observation(data=obs_data)

    def step(self, action_idx):
        reward = 0.0
        done = False
        info = {"explanation": ""}
        
        if not self.task_queue:
            return self._get_observation(), -5.0, True, {"msg": "No tasks left", "explanation": "No tasks available to schedule."}

        # Dynamic Task Arrival
        if random.random() < 0.15:
            self._generate_task()

        # Hybrid Decision System: Rule-based Validator
        # Check if any task is starving
        starving_indices = [i for i, t in enumerate(self.task_queue) if t.waiting_time > max(15, 2 * t.burst_time)]
        
        # Action Handling & Validation
        if action_idx < 0 or action_idx >= len(self.task_queue):
            reward -= 5.0 
            actual_action = 0 
            info["explanation"] += "Invalid task index provided. Defaulted to task 0. "
        else:
            actual_action = action_idx
            # Fairness Check: If there are starving tasks and we pick a non-starving one
            if starving_indices and actual_action not in starving_indices:
                reward -= 10.0 # Starvation avoidance penalty
                info["explanation"] += f"Fairness Alert: Ignored starving task(s) {starving_indices}. "
            else:
                info["explanation"] += "Action validated by hybrid controller. "

        # Execute Task
        task = self.task_queue[actual_action]
        task.status = TaskStatus.RUNNING
        task.remaining_time -= 1
        self.current_time += 1
        
        # Update waiting times for others
        for i, t in enumerate(self.task_queue):
            if i != actual_action:
                t.waiting_time += 1
                t.status = TaskStatus.READY
        
        # Adaptive Reward System
        # 1. Waiting penalty (weighted by priority)
        waiting_penalty = sum(t.waiting_time * 0.05 * t.priority for t in self.task_queue)
        
        # 2. Completion bonus
        completion_bonus = 0.0
        if task.remaining_time <= 0:
            completion_bonus = 10.0 * task.priority
            task.status = TaskStatus.COMPLETED
            self.finished_tasks.append(task)
            self.task_queue.pop(actual_action)
            info["explanation"] += f"Task {task.task_id} completed (Priority: {task.priority}). "
        else:
            info["explanation"] += f"Task {task.task_id} is running (Remaining: {task.remaining_time}). "

        # 3. Utilization bonus (if queue not empty)
        util_bonus = 1.0 if self.task_queue else 0.0

        reward += (completion_bonus - waiting_penalty + util_bonus)
            
        if not self.task_queue:
            done = True
            
        self.total_reward += reward
        return self._get_observation(), float(reward), done, info

# API
env = CPUSchedulerEnv()

@app.get("/", include_in_schema=False)
def index():
    return FileResponse("index.html")

@app.post("/reset", response_model=ResetResponse)
def reset():
    obs = env.reset()
    return ResetResponse(observation=obs)

@app.post("/step", response_model=StepResponse)
def step(action: Action):
    obs, reward, done, info = env.step(action.task_index)
    explanation = info.get("explanation", "")
    return StepResponse(observation=obs, reward=reward, done=done, info=info, explanation=explanation)

@app.get("/state", response_model=StateResponse)
def get_state():
    return StateResponse(state=env.state())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

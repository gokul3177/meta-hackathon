from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class TaskStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"

class Task(BaseModel):
    task_id: int
    arrival_time: int
    burst_time: int
    remaining_time: int
    waiting_time: int = 0
    priority: int = 5  # 1 (low) to 10 (high)
    status: TaskStatus = TaskStatus.READY

class EnvState(BaseModel):
    current_time: int
    task_queue: List[Task]
    finished_tasks: List[Task] = []
    cpu_utilization: float = 0.0
    avg_waiting_time: float = 0.0
    total_reward: float = 0.0

class Action(BaseModel):
    task_index: int

class Observation(BaseModel):
    # Fixed-size list representation for RL
    # Each task is [remaining_time, waiting_time, priority]
    data: List[float]

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetResponse(BaseModel):
    observation: Observation

class StateResponse(BaseModel):
    state: EnvState

from pydantic import BaseModel
from typing import List, Optional

class Task(BaseModel):
    task_id: int
    arrival_time: int
    burst_time: int
    remaining_time: int
    waiting_time: int = 0

class EnvState(BaseModel):
    current_time: int
    task_queue: List[Task]
    finished_tasks: List[Task] = []

class Action(BaseModel):
    task_index: int

class Observation(BaseModel):
    # Fixed-size list representation for RL
    # Each task is [remaining_time, waiting_time]
    # Padded with 0s if queue is smaller than max_tasks
    data: List[float]

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

class ResetResponse(BaseModel):
    observation: Observation

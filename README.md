# 🧠 AI-Based CPU Task Scheduler (OpenEnv)

A complete, real-world Reinforcement Learning environment for optimizing CPU task scheduling, built for the **Meta PyTorch OpenEnv Hackathon**.

![Status](https://img.shields.io/badge/Status-Complete-success)
![Platform](https://img.shields.io/badge/Platform-OpenEnv-blue)
![Lang](https://img.shields.io/badge/Python-3.10%2B-blue)

## 🎯 Project Goals
Build a standardized RL environment that follows the **OpenEnv API** (`step`, `reset`, `state`). This simulator models a priority-based CPU scheduler where an AI agent must manage dynamic task arrivals and conflicting priorities to minimize system latency.

## 🏗️ Advanced Environment Design

### 1. State Space (`state()`)
The environment provides a full `EnvState` object containing:
* **Task Queue**: Detailed list of `ready` tasks with remaining time and priority.
* **CPU Load**: Real-time utilization metrics.
* **System Wait Time**: Cumulative and average waiting times.

### 2. Observation Space
A fixed-size padded vector suitable for PyTorch models:
* `[remaining_time, waiting_time, priority]` for each of the top 10 task slots.

### 3. Reward Function (Real-World Weighted)
* **Time Penalty**: `-1.0` per step.
* **Wait Penalty**: Weighted by priority (Higher priority tasks waiting = more negative reward).
* **Completion Bonus**: `+10.0 * TaskPriority`.

## 🚀 How to Run Locally

### Prerequisites
* Python 3.10+
* `pip install -r requirements.txt`

### Run Advanced Inference
Execute the built-in inference script to see the **Priority-Aware SJF** policy:
```bash
python inference.py
```

### Start OpenEnv API Server
```bash
uvicorn env:app --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/state` to see the live system metrics.

## 🧪 Example Output
```text
[01] Action: Task 1 (Prio: 8) | Reward:  -1.30 | Done: False
[02] Action: Task 1 (Prio: 8) | Reward:  78.70 | Done: False
...
📊 REAL-WORLD PERFORMANCE REPORT
Total Time Units    : 28
Total Cumulative Reward: 142.40
Average Waiting Time : 8.50
Tasks Completed     : 5
Final CPU Utilization: 100.0%
```

## 🐳 Deployment
Optimized for **Hugging Face Spaces** (Docker SDK). The server runs on port `7860` by default in the container.

## 📄 License
MIT License

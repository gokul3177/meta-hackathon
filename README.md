# 🧠 AI-Based CPU Task Scheduler (OpenEnv)

A minimal, high-performance Reinforcement Learning environment for optimizing CPU task scheduling, built for the **Meta PyTorch OpenEnv Hackathon**.

![Status](https://img.shields.io/badge/Status-Working-success)
![Platform](https://img.shields.io/badge/Platform-OpenEnv-blue)
![Lang](https://img.shields.io/badge/Python-3.10%2B-blue)

## 🎯 Problem Statement
CPU scheduling is a fundamental operating system task. Traditional algorithms like **First-In-First-Out (FIFO)** or **Round Robin (RR)** often fall short in dynamic workloads. This project provides a simulator where an RL agent can learn to minimize **Average Waiting Time** and maximize system efficiency by selecting which task to execute at every time unit.

## 🏗️ RL Formulation

### 1. State Space
The environment maintains a queue of tasks. The observation provided to the agent is a fixed-size vector representing:
* **Remaining Time**: Time units left for the task to complete.
* **Waiting Time**: Time units the task has spent in the queue.

### 2. Action Space
* **Discrete(N)**: The index of the task in the queue to be executed for 1 time unit.

### 3. Reward Function
* `-1.0`: Penalty per time step (encourages speed).
* `+10.0`: Bonus for completing a task.
* `-5.0`: Penalty for idleness if tasks are available.

## 🚀 How to Run Locally

### Prerequisites
* Python 3.10+
* `pip install -r requirements.txt`

### Run Inference (Baseline SJF)
Run the built-in inference script to see a Shortest Job First (SJF) policy in action:
```bash
python inference.py
```

### Start environment API
The environment is exposed via FastAPI, allowing external agents to interact with it:
```bash
uvicorn env:app --host 0.0.0.0 --port 8000
```

## 🧪 Example Output
```text
Step 1: Action (Task 2), Reward: -1.00, Done: False
Step 2: Action (Task 2), Reward: 9.00, Done: False
...
📊 PERFORMANCE SUMMARY
Total Steps (Time Units): 24
Total Reward: 26.00
Average Waiting Time: 7.40
Tasks Completed: 5
```

## 🐳 Deployment (Hugging Face Spaces)
This repository is ready for deployment as a **Docker Space**.
1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Push these files to the repository.
4. Your environment API will be available at your Space's URL.

## 📄 License
MIT License

from env import CPUSchedulerEnv
import numpy as np

def run_inference():
    print("🚀 Starting CPU Scheduler RL Inference...")
    print("-" * 40)
    
    env = CPUSchedulerEnv()
    obs = env.reset()
    
    total_reward = 0
    done = False
    steps = 0
    
    # Store initial task list for baseline logging
    initial_tasks = [(t.task_id, t.burst_time) for t in env.task_queue]
    print(f"Initial Tasks: {initial_tasks}")
    
    while not done:
        # Simple Policy: Shortest Remaining Time First (SRTF)
        # Find index of task with minimum remaining time
        current_queue = env.task_queue
        if not current_queue:
            break
            
        remaining_times = [t.remaining_time for t in current_queue]
        action = int(np.argmin(remaining_times))
        
        task_id = current_queue[action].task_id
        
        # Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}: Action (Task {task_id}), Reward: {reward:.2f}, Done: {done}")
        
    print("-" * 40)
    print("📊 PERFORMANCE SUMMARY")
    print(f"Total Steps (Time Units): {env.current_time}")
    print(f"Total Reward: {total_reward:.2f}")
    
    # Calculate Average Waiting Time
    waiting_times = [t.waiting_time for t in env.finished_tasks]
    avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
    
    print(f"Average Waiting Time: {avg_waiting_time:.2f}")
    print(f"Tasks Completed: {len(env.finished_tasks)}")
    print("-" * 40)
    print("✅ Inference completed successfully.")

if __name__ == "__main__":
    run_inference()

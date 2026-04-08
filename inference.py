from env import CPUSchedulerEnv
import numpy as np

def run_inference():
    print("🚀 Starting Advanced OpenEnv CPU Scheduler Inference...")
    print("-" * 50)
    
    env = CPUSchedulerEnv()
    obs = env.reset()
    
    total_reward = 0
    done = False
    steps = 0
    
    while not done:
        # HP-SJF Policy: Select task based on Highest Priority, then Shortest Remaining Time
        current_queue = env.task_queue
        if not current_queue:
            break
            
        # Decision logic: Calculate a "Score" for each task
        # Score = Priority / RemainingTime
        # Higher score = Better task to pick
        scores = []
        for t in current_queue:
            # Avoid division by zero, though remaining_time should be > 0
            score = t.priority / (t.remaining_time if t.remaining_time > 0 else 0.1)
            scores.append(score)
            
        action = int(np.argmax(scores))
        task = current_queue[action]
        
        # Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"[{steps:02d}] Action: Task {task.task_id} (Prio: {task.priority}) | Reward: {reward:6.2f} | Done: {done}")
        
    print("-" * 50)
    print("📊 REAL-WORLD PERFORMANCE REPORT")
    final_state = env.state()
    print(f"Total Time Units    : {final_state.current_time}")
    print(f"Total Cumulative Reward: {final_state.total_reward:.2f}")
    print(f"Average Waiting Time : {final_state.avg_waiting_time:.2f}")
    print(f"Tasks Completed     : {len(final_state.finished_tasks)}")
    print(f"Final CPU Utilization: {final_state.cpu_utilization * 100:.1f}%")
    print("-" * 50)
    print("✅ Advanced Inference completed successfully.")

if __name__ == "__main__":
    run_inference()

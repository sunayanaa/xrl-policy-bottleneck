# ==============================================================================
# Program Name: evaluate_distilled_policy.py
# 
# Description: 
# This script performs the final empirical evaluation of the distilled CART 
# tree policy. It loads the saved tree model and executes it inside the 
# MuJoCo HalfCheetah-v4 environment over multiple episodes to measure its 
# true functional competence (average cumulative reward).
# ==============================================================================

# --- COLAB SETUP ---
!pip install -q scikit-learn joblib gymnasium[mujoco]

import os
import numpy as np
import joblib
import gymnasium as gym
from google.colab import drive
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Mount Google Drive to access the saved model
drive.mount('/content/drive')

def evaluate_tree(env_id="HalfCheetah-v4", max_depth=14, num_episodes=10):
    """
    Loads the distilled tree and evaluates its performance in the environment.
    """
    # 1. Setup paths
    drive_path = '/content/drive/My Drive/XRL_Experiments/MuJoCo'
    model_file = os.path.join(drive_path, f'distilled_tree_depth_{max_depth}.joblib')

    print(f"--- Starting Empirical Evaluation for {env_id} ---")
    
    # 2. Load the Distilled Tree
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found at {model_file}.")
        
    print(f"Loading distilled tree from: {model_file}")
    tree = joblib.load(model_file)
    
    # 3. Initialize the Environment
    env = gym.make(env_id)
    episode_rewards = []

    print(f"\nRunning {num_episodes} evaluation episodes...")

    # 4. Evaluation Loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            # Reshape observation for sklearn (1 sample, n features)
            obs_reshaped = obs.reshape(1, -1)
            
            # The tree predicts both Actions and Values [A, V]
            # We only need the actions (the first 6 dimensions for HalfCheetah)
            prediction = tree.predict(obs_reshaped)[0]
            action = prediction[:6] 
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            done = terminated or truncated
            
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f} (Steps = {step_count})")

    # 5. Calculate Final Metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("\n--- Final Evaluation Results ---")
    print(f"Tree Depth: {max_depth}")
    print(f"Mean Cumulative Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Execute the evaluation
evaluate_tree(max_depth=14, num_episodes=10)
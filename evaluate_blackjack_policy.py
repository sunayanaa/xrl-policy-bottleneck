# ==============================================================================
# Program Name: evaluate_blackjack_policy.py
# 
# Description: 
# Evaluates the distilled CART tree policy inside the Gymnasium Blackjack-v1 
# environment over 100,000 episodes to determine its true functional competence, 
# win rate, and average reward.
# ==============================================================================

# --- COLAB SETUP ---
!pip install -q scikit-learn joblib gymnasium

import os
import numpy as np
import joblib
import gymnasium as gym
from google.colab import drive
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Mount Google Drive
drive.mount('/content/drive')

# --- THE FIX: Custom Wrapper to flatten the Tuple space ---
class BlackjackWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=32.0, shape=(3,), dtype=np.float32)

    def observation(self, obs):
        return np.array(obs, dtype=np.float32)

def evaluate_discrete_tree(max_depth=7, num_episodes=100000):
    # 1. Setup paths
    drive_path = '/content/drive/My Drive/XRL_Experiments/Blackjack'
    model_file = os.path.join(drive_path, f'blackjack_tree_depth_{max_depth}.joblib')

    print("--- Starting Empirical Evaluation (Blackjack) ---")
    
    # 2. Load the Distilled Tree
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found at {model_file}.")
        
    print(f"Loading distilled tree from: {model_file}")
    tree = joblib.load(model_file)
    
    # 3. Initialize the Environment
    env = BlackjackWrapper(gym.make("Blackjack-v1"))
    
    rewards = []
    wins = 0
    losses = 0
    draws = 0

    print(f"\nRunning {num_episodes} evaluation hands (this takes ~10 seconds)...")

    # 4. Evaluation Loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # Reshape observation for sklearn (1 sample, n features)
            obs_reshaped = obs.reshape(1, -1)
            
            # The tree predicts [Prob_Stick, Prob_Hit, Value]
            # We take the argmax of the first 2 dimensions (the action probabilities)
            prediction = tree.predict(obs_reshaped)[0]
            action = np.argmax(prediction[:2])
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            done = terminated or truncated
            
        rewards.append(total_reward)
        
        # Track Win/Loss/Draw
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1
            
        if (episode + 1) % 25000 == 0:
            print(f"Played {episode + 1} hands...")

    # 5. Calculate Final Metrics
    mean_reward = np.mean(rewards)
    win_rate = (wins / num_episodes) * 100
    loss_rate = (losses / num_episodes) * 100
    draw_rate = (draws / num_episodes) * 100

    print("\n--- Final Evaluation Results ---")
    print(f"Tree Depth: {max_depth}")
    print(f"Total Hands Played: {num_episodes:,}")
    print(f"Mean Reward per Hand: {mean_reward:.4f}")
    print(f"Win Rate:  {win_rate:.2f}%")
    print(f"Loss Rate: {loss_rate:.2f}%")
    print(f"Draw Rate: {draw_rate:.2f}%")

# Execute the evaluation
evaluate_discrete_tree()
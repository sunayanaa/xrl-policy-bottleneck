# ==============================================================================
# Program Name: generate_blackjack_dataset.py
# 
# Description: 
# This script trains an expert PPO policy for the Gymnasium Blackjack-v1 
# environment. It perfectly satisfies the reviewer's request for a "card game" 
# to prove generalization to discrete action spaces.
# It generates a dataset of states, discrete actions (0=Stick, 1=Hit), 
# and strategic state-values V(s) for Value-Aware Policy Distillation.
# ==============================================================================

# --- COLAB SETUP ---
!pip install -q stable-baselines3[extra] gymnasium

import os
import numpy as np
import torch
import gymnasium as gym
from google.colab import drive
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Mount Google Drive
drive.mount('/content/drive')

# --- THE FIX: Custom Wrapper to flatten the Tuple space for SB3 ---
class BlackjackWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Convert Tuple(32, 11, 2) into a simple flat Box of 3 floats
        self.observation_space = gym.spaces.Box(low=0.0, high=32.0, shape=(3,), dtype=np.float32)

    def observation(self, obs):
        # Convert the (player_sum, dealer_card, usable_ace) tuple to a numpy array
        return np.array(obs, dtype=np.float32)

def train_and_collect_blackjack(num_train_steps=100000, num_collect_steps=50000):
    # 1. Setup Save Directories
    save_dir = '/content/drive/My Drive/XRL_Experiments/Blackjack'
    os.makedirs(save_dir, exist_ok=True)
    dataset_file = os.path.join(save_dir, 'blackjack_expert_dataset.npz')
    model_file = os.path.join(save_dir, 'ppo_blackjack_expert.zip')

    print("--- Phase 1: Training PPO Expert on Blackjack ---")
    
    # 2. Setup standard Gymnasium Environment with our Custom Wrapper
    env = BlackjackWrapper(gym.make("Blackjack-v1"))

    # Train the model
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-3)
    print(f"Training PPO for {num_train_steps} timesteps (this takes ~30 seconds)...")
    model.learn(total_timesteps=num_train_steps)
    model.save(model_file)
    print(f"Expert model trained and saved to {model_file}")

    print("\n--- Phase 2: Collecting Expert Rollouts ---")
    
    obs, _ = env.reset()
    states = []
    actions = []
    values = []

    print(f"Collecting {num_collect_steps} timesteps of expert rollouts...")

    # 3. Rollout Loop
    for step in range(num_collect_steps):
        states.append(obs)
        
        # Get discrete action from the trained expert (0: Stick, 1: Hit)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        
        # Extract the Value V(s)
        obs_tensor = torch.tensor(obs).unsqueeze(0).to(model.device)
        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor).cpu().numpy()[0][0]
        values.append(value)
        
        # Step the environment forward
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        if (step + 1) % 10000 == 0:
            print(f"Collected {step + 1} / {num_collect_steps} steps...")

    # 4. Convert to optimized numpy arrays
    states_np = np.array(states)
    actions_np = np.array(actions)
    values_np = np.array(values)

    # 5. Save the dataset
    np.savez_compressed(
        dataset_file, 
        states=states_np, 
        actions=actions_np, 
        values=values_np
    )

    print("\n--- Data Collection Complete ---")
    print(f"Dataset successfully saved to: {dataset_file}")
    print(f"States shape:  {states_np.shape} (Player Sum, Dealer Card, Usable Ace)")
    print(f"Actions shape: {actions_np.shape} (Discrete: 0=Stick, 1=Hit)")
    print(f"Values shape:  {values_np.shape}")

# Execute
train_and_collect_blackjack()

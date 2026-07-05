# ==============================================================================
# Program Name: generate_cartpole_dataset.py
# 
# Description: 
# This script trains an expert PPO policy for the Gymnasium CartPole-v1 
# environment. It satisfies the reviewer's request for an additional classic, 
# low-dimensional, discrete control benchmark to complement HalfCheetah 
# (continuous) and Blackjack (discrete, stochastic).
# It generates a dataset of states, discrete actions (0=Push Left, 1=Push 
# Right), and strategic state-values V(s) for Value-Aware Policy Distillation.
# ==============================================================================
# Runs on CPU
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

def train_and_collect_cartpole(num_train_steps=100000, num_collect_steps=50000):
    # 1. Setup Save Directories
    save_dir = '/content/drive/My Drive/paper/XRL_Experiments/CartPole'
    os.makedirs(save_dir, exist_ok=True)
    dataset_file = os.path.join(save_dir, 'cartpole_expert_dataset.npz')
    model_file = os.path.join(save_dir, 'ppo_cartpole_expert.zip')

    print("--- Phase 1: Training PPO Expert on CartPole ---")

    # 2. Setup standard Gymnasium Environment (native flat Box(4,) observation,
    #    no wrapper needed unlike Blackjack's Tuple space)
    env = gym.make("CartPole-v1")

    # Train the model
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-3)
    print(f"Training PPO for {num_train_steps} timesteps (this takes ~2 mins)...")
    model.learn(total_timesteps=num_train_steps)
    model.save(model_file)
    print(f"Expert model trained and saved to {model_file}")

    # 3. Quick sanity check on the trained expert before collecting data
    eval_rewards = []
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        eval_rewards.append(ep_reward)
    print(f"Expert sanity check: {np.mean(eval_rewards):.1f} +/- {np.std(eval_rewards):.1f} "
          f"(max possible = 500.0)")
    if np.mean(eval_rewards) < 400:
        print("WARNING: Expert has not solved CartPole reliably. Consider increasing "
              "num_train_steps before proceeding.")

    print("\n--- Phase 2: Collecting Expert Rollouts ---")

    obs, _ = env.reset()
    states = []
    actions = []
    values = []

    print(f"Collecting {num_collect_steps} timesteps of expert rollouts...")

    # 4. Rollout Loop
    for step in range(num_collect_steps):
        states.append(obs)

        # Get discrete action from the trained expert (0: Left, 1: Right)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)

        # Extract the Value V(s)
        obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(model.device)
        with torch.no_grad():
            value = model.policy.predict_values(obs_tensor).cpu().numpy()[0][0]
        values.append(value)

        # Step the environment forward
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

        if (step + 1) % 10000 == 0:
            print(f"Collected {step + 1} / {num_collect_steps} steps...")

    # 5. Convert to optimized numpy arrays
    states_np = np.array(states)
    actions_np = np.array(actions)
    values_np = np.array(values)

    # 6. Save the dataset
    np.savez_compressed(
        dataset_file,
        states=states_np,
        actions=actions_np,
        values=values_np
    )

    print("\n--- Data Collection Complete ---")
    print(f"Dataset successfully saved to: {dataset_file}")
    print(f"States shape:  {states_np.shape} (Cart Position, Cart Velocity, "
          f"Pole Angle, Pole Angular Velocity)")
    print(f"Actions shape: {actions_np.shape} (Discrete: 0=Push Left, 1=Push Right)")
    print(f"Values shape:  {values_np.shape}")

# Execute
train_and_collect_cartpole()

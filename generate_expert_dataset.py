# ==============================================================================
# Program Name: generate_expert_dataset.py
# 
# Description: 
# This script is designed to run in a Google Colab environment. It automates 
# the process of collecting an expert rollout dataset for the "Value-Aware 
# Policy Distillation" framework presented in the manuscript.
# 
# Specifically, it performs the following operations:
# 1. Mounts Google Drive to provide persistent storage for the generated data.
# 2. Downloads a pre-trained expert Proximal Policy Optimization (PPO) model 
#    for the MuJoCo HalfCheetah-v4 continuous control environment directly 
#    from the Stable Baselines3 Hugging Face repository.
# 3. Executes the expert policy in the environment to collect a specified 
#    number of timesteps (default: 50,000).
# 4. Records the continuous environment states (S), the deterministic expert 
#    motor torque actions (A), and the strategic state-values V(s) estimated 
#    by the PPO expert's critic network.
# 5. Saves the extracted (S, A, V) dataset as a compressed '.npz' array 
#    directly into the mounted Google Drive. This dataset is subsequently 
#    used to train the capacity-constrained CART tree student policy.
# ==============================================================================

# --- COLAB SETUP AND DRIVE MOUNTING ---
# These commands install the required libraries in the Colab environment
!pip install -q stable-baselines3[extra] rl_zoo3 gymnasium[mujoco] huggingface_hub

from google.colab import drive
import os
import numpy as np
import torch
import gymnasium as gym
from huggingface_hub import hf_hub_download
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Mount Google Drive to save the dataset permanently
drive.mount('/content/drive')

def collect_expert_rollouts(env_id="HalfCheetah-v4", num_steps=50000):
    """
    Downloads a pre-trained expert, runs rollouts, and extracts (S, A, V) tuples.
    Saves the output directly to Google Drive.
    """
    # 1. Ensure save directory exists in your Google Drive
    save_dir = '/content/drive/My Drive/paper/XRL_Experiments/MuJoCo'
    os.makedirs(save_dir, exist_ok=True)
    dataset_file = os.path.join(save_dir, 'halfcheetah_expert_dataset.npz')

    print(f"--- Starting Expert Data Collection for {env_id} ---")
    print(f"Data will be saved to: {dataset_file}")
    
    # 2. Download the official pre-trained PPO expert from Hugging Face
    repo_id = "sb3/ppo-HalfCheetah-v3"
    filename = "ppo-HalfCheetah-v3.zip"
    print("\nDownloading expert model weights from Hugging Face...")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    
    # Load the model
    teacher_model = PPO.load(model_path)
    device = teacher_model.device
    print(f"Expert PPO Model loaded successfully on device: {device}\n")

    # 3. Initialize the environment
    env = gym.make(env_id)
    obs, _ = env.reset()

    # Data storage lists
    states = []
    actions = []
    values = []

    print(f"Collecting {num_steps} timesteps of expert rollouts...")

    # 4. Rollout Loop
    for step in range(num_steps):
        # Record the current state
        states.append(obs)
        
        # Get the deterministic action from the expert policy
        action, _ = teacher_model.predict(obs, deterministic=True)
        actions.append(action)
        
        # Extract the Value V(s) using the PPO critic network
        # Convert the observation to a tensor for the network forward pass
        obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            value = teacher_model.policy.predict_values(obs_tensor).cpu().numpy()[0][0]
        values.append(value)
        
        # Step the environment forward
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
            
        if (step + 1) % 10000 == 0:
            print(f"Collected {step + 1} / {num_steps} steps...")

    # 5. Convert lists to optimized numpy arrays
    states_np = np.array(states)
    actions_np = np.array(actions)
    values_np = np.array(values)

    # 6. Save the dataset to an NPZ archive in Google Drive
    np.savez_compressed(
        dataset_file, 
        states=states_np, 
        actions=actions_np, 
        values=values_np
    )

    print("\n--- Data Collection Complete ---")
    print(f"Dataset successfully saved to Google Drive at: {dataset_file}")
    print(f"States shape:  {states_np.shape}")
    print(f"Actions shape: {actions_np.shape}")
    print(f"Values shape:  {values_np.shape}")

# Execute the collection
collect_expert_rollouts()
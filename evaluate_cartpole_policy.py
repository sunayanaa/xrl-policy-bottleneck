# ==============================================================================
# Program Name: evaluate_cartpole_policy.py
# 
# Description: 
# Evaluates the distilled CART tree policy inside the Gymnasium CartPole-v1 
# environment over 100 episodes to determine its true functional competence 
# and average reward. Uses mean +/- std reporting to match the convention 
# already used for HalfCheetah and Blackjack in the generalization table.
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

def evaluate_discrete_tree(max_depth=6, num_episodes=100):
    # 1. Setup paths
    drive_path = '/content/drive/My Drive/paper/XRL_Experiments/CartPole'
    model_file = os.path.join(drive_path, f'cartpole_tree_depth_{max_depth}.joblib')

    print("--- Starting Empirical Evaluation (CartPole) ---")

    # 2. Load the Distilled Tree
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found at {model_file}.")

    print(f"Loading distilled tree from: {model_file}")
    tree = joblib.load(model_file)

    # 3. Initialize the Environment (native flat Box(4,) observation, no
    #    wrapper needed unlike Blackjack's Tuple space)
    env = gym.make("CartPole-v1")

    rewards = []
    solved_count = 0  # episodes reaching the max 500-step cap

    print(f"\nRunning {num_episodes} evaluation episodes...")

    # 4. Evaluation Loop
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Reshape observation for sklearn (1 sample, n features)
            obs_reshaped = obs.reshape(1, -1)

            # The tree predicts [Prob_Left, Prob_Right, Value]
            # We take the argmax of the first 2 dimensions (the action probabilities)
            prediction = tree.predict(obs_reshaped)[0]
            action = int(np.argmax(prediction[:2]))

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            done = terminated or truncated

        rewards.append(total_reward)
        if total_reward >= 500:
            solved_count += 1

        if (episode + 1) % 25 == 0:
            print(f"Completed {episode + 1} episodes...")

    # 5. Calculate Final Metrics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    solved_rate = (solved_count / num_episodes) * 100

    print("\n--- Final Evaluation Results ---")
    print(f"Tree Depth: {max_depth}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Mean Reward per Episode: {mean_reward:.2f} +/- {std_reward:.2f}  (max = 500.0)")
    print(f"Episodes Reaching Max Steps (500): {solved_count} ({solved_rate:.1f}%)")

# Execute the evaluation
evaluate_discrete_tree()

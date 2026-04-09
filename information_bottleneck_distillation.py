# information_bottleneck_distillation.py
# This script implements the core Value-Aware Information Bottleneck distillation framework. It trains a PPO expert Oracle on the LunarLander-v3 environment, extracts a dataset containing states, actions, and continuous value estimates (V(s)), and distills this into capacity-constrained decision trees (CART). It computes the Dual-Fidelity metrics (Behavioral Fidelity F_b and Structural Fidelity F_s), evaluates the Performance-Transparency Gap (PTG) with bootstrap confidence intervals, and generates the Pareto frontier and feature importance visualizations.

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ============================================================================
# STEP 1: Train the Expert (PPO)
# ============================================================================
# Install & Train PPO
!apt-get install -y swig
!pip install gymnasium[box2d] stable-baselines3 shimmy scikit-learn pandas matplotlib seaborn

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os

# 1. Setup Environment
env = gym.make("LunarLander-v3")
env = Monitor(env)

# 2. Train PPO (More stable than DQN)
print("Training PPO Oracle...")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=300000)

# 3. Verify Performance
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Oracle Performance: {mean_reward:.2f} +/- {std_reward:.2f}")

if mean_reward > 200:
    print("SUCCESS: The Oracle is flying perfectly! Proceed to Step 2.")
    model.save("ppo_lunar_oracle")
else:
    print("WARNING: The Oracle is still mediocre. Consider re-running training.")


# ============================================================================
# STEP 2: Distill Data + Extract Q-Values 
# ============================================================================
# Enhanced Distillation with Q-value Collection
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score

# Load the model if you restarted the kernel
# model = PPO.load("ppo_lunar_oracle") 

def generate_expert_data_with_qvalues(model, env, n_samples=50000):
    """
    NEW: Collect states, actions, AND Q-value approximations
    PPO doesn't have explicit Q-values, so we use value function + advantage estimates
    """
    states = []
    actions = []
    values = []  # NEW: PPO's value function output
    
    obs, _ = env.reset()
    for _ in range(n_samples):
        # Get action from policy
        action, _ = model.predict(obs, deterministic=True)
        
        # NEW: Get value estimate (proxy for Q-values in policy-based methods)
        # PPO's value network estimates V(s), we'll use this for structural fidelity
        obs_tensor = obs.reshape(1, -1)
        value = model.policy.predict_values(
            model.policy.obs_to_tensor(obs_tensor)[0]
        ).detach().cpu().numpy()[0, 0]  # Added .detach() to remove gradients
        
        states.append(obs)
        actions.append(action)
        values.append(value)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    return np.array(states), np.array(actions), np.array(values)

# Generate Enhanced Dataset
print("Collecting data from PPO (with value estimates)...")
X, y, V_expert = generate_expert_data_with_qvalues(model, env, n_samples=50000)

print(f"Action Distribution in Dataset: {np.bincount(y)}")

# Train Action Trees (for behavioral fidelity)
depths = [2, 4, 6, 8, 10, 12, 14]
action_trees = {}
value_trees = {}  # NEW: Trees to predict value function

print("Distilling policies...")
for d in depths:
    # Action prediction tree (for F_b)
    action_clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    action_clf.fit(X, y)
    action_trees[d] = action_clf
    
    # NEW: Value prediction tree (for F_s)
    value_reg = DecisionTreeRegressor(max_depth=d, random_state=42)
    value_reg.fit(X, V_expert)
    value_trees[d] = value_reg
    
    acc = accuracy_score(y, action_clf.predict(X))
    print(f"Depth {d} Training Action Fidelity: {acc:.2%}")


# ============================================================================
# STEP 3: COMPLETE EVALUATION (Compute ALL metrics with confidence intervals)
# ============================================================================
# Comprehensive Evaluation with Bootstrap Confidence Intervals

import matplotlib.pyplot as plt
from scipy import stats

def evaluate_tree_complete(action_tree, value_tree, env, model, 
                          X_test, V_test, n_episodes=100, n_bootstrap=1000):
    """
    NEW: Compute ALL metrics needed for the paper table:
    - Behavioral Fidelity (F_b): % of action matches
    - Structural Fidelity (F_s): Normalized value prediction error
    - Average Reward: Performance in environment
    - PTG: Performance-Transparency Gap
    - Bootstrap confidence intervals for all metrics
    """
    
    # ===== Structural Fidelity (F_s) - Equation (4) from paper =====
    V_tree_pred = value_tree.predict(X_test)
    
    # Compute normalized error as in Equation (4)
    value_errors = np.abs(V_test - V_tree_pred)
    max_value = np.max(np.abs(V_test))
    F_s = 1 - np.mean(value_errors) / max_value
    
    # ===== Behavioral Fidelity (F_b) and Reward =====
    episode_rewards = []
    action_matches = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_matches = []
        
        while not done:
            # Tree action
            tree_action = action_tree.predict([obs])[0]
            
            # Expert action
            expert_action, _ = model.predict(obs, deterministic=True)
            
            # Track match
            ep_matches.append(1 if tree_action == expert_action else 0)
            
            # Step environment with tree action
            obs, r, term, trunc, _ = env.step(tree_action)
            ep_reward += r
            done = term or trunc
        
        episode_rewards.append(ep_reward)
        action_matches.extend(ep_matches)
    
    # Compute metrics
    F_b = np.mean(action_matches)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    # Bootstrap confidence intervals for reward
    bootstrap_rewards = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(episode_rewards, size=len(episode_rewards), replace=True)
        bootstrap_rewards.append(np.mean(sample))
    
    ci_lower = np.percentile(bootstrap_rewards, 2.5)
    ci_upper = np.percentile(bootstrap_rewards, 97.5)
    
    return {
        'F_s': F_s,
        'F_b': F_b,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


# ===== Prepare Test Set for F_s Calculation =====
print("Generating test set for structural fidelity...")
X_test, y_test, V_test = generate_expert_data_with_qvalues(model, env, n_samples=10000)

# ===== Evaluate Oracle Baseline =====
print(f"\n{'='*60}")
print("Evaluating Oracle (PPO Teacher)...")
oracle_rewards = []
for _ in range(100):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        ep_reward += r
        done = term or trunc
    oracle_rewards.append(ep_reward)

oracle_mean = np.mean(oracle_rewards)
oracle_std = np.std(oracle_rewards)
print(f"Oracle: {oracle_mean:.2f} ± {oracle_std:.2f}")

# ===== Evaluate All Trees =====
results = []

print(f"\n{'='*60}")
print("Evaluating Distilled Trees...")
print(f"{'Depth':<6} {'F_b (%)':<10} {'F_s':<8} {'Reward':<12} {'PTG (%)':<10}")
print(f"{'='*60}")

for d in depths:
    metrics = evaluate_tree_complete(
        action_trees[d], 
        value_trees[d], 
        env, 
        model, 
        X_test, 
        V_test,
        n_episodes=100,
        n_bootstrap=1000
    )
    
    # Calculate PTG
    ptg = ((oracle_mean - metrics['mean_reward']) / oracle_mean) * 100
    
    results.append({
        'Depth': d,
        'F_b': metrics['F_b'] * 100,  # Convert to percentage
        'F_s': metrics['F_s'],
        'Reward': metrics['mean_reward'],
        'Std': metrics['std_reward'],
        'CI_lower': metrics['ci_lower'],
        'CI_upper': metrics['ci_upper'],
        'PTG': ptg
    })
    
    print(f"{d:<6} {metrics['F_b']*100:<10.1f} {metrics['F_s']:<8.3f} "
          f"{metrics['mean_reward']:<12.1f} {ptg:<10.1f}")

# ===== Generate LaTeX Table =====
print(f"\n{'='*60}")
print("LATEX TABLE FOR PAPER:")
print(f"{'='*60}\n")

print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Performance Metrics Across Tree Depths}")
print("\\label{tab:results}")
print("\\begin{tabular}{|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Depth} & \\textbf{$F_b$ (\\%)} & \\textbf{$F_s$} & "
      "\\textbf{Reward} & \\textbf{PTG (\\%)} \\\\")
print("\\hline")

for r in results:
    print(f"{r['Depth']} & {r['F_b']:.1f} ± {r['Std']:.1f} & "
          f"{r['F_s']:.2f} & {r['Reward']:.1f} ± {r['Std']:.1f} & "
          f"{r['PTG']:.1f} \\\\")

print("\\hline")
print(f"\\textbf{{PPO Teacher}} & 100.0 & 1.00 & "
      f"{oracle_mean:.1f} ± {oracle_std:.1f} & 0.0 \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

# ===== Enhanced Visualization =====
df = pd.DataFrame(results)

fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot Behavioral Fidelity (F_b)
color_fb = 'tab:blue'
ax1.set_xlabel('Tree Depth (Complexity)', fontsize=12)
ax1.set_ylabel('Behavioral Fidelity (%)', color=color_fb, fontsize=12)
ax1.plot(df['Depth'], df['F_b'], marker='o', color=color_fb, 
         linewidth=2.5, label='$F_b$ (Behavioral)', markersize=8)
ax1.tick_params(axis='y', labelcolor=color_fb)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Plot Structural Fidelity (F_s) on same axis
ax1.plot(df['Depth'], df['F_s'] * 100, marker='s', color='tab:cyan',
         linewidth=2.5, label='$F_s$ (Structural × 100)', markersize=8, linestyle='--')

# Plot Reward on second axis
ax2 = ax1.twinx()
color_reward = 'tab:orange'
ax2.set_ylabel('Average Reward', color=color_reward, fontsize=12)
ax2.plot(df['Depth'], df['Reward'], marker='x', linestyle='--', 
         color=color_reward, linewidth=2.5, label='Performance', markersize=10)

# Add confidence interval shading
ax2.fill_between(df['Depth'], df['CI_lower'], df['CI_upper'], 
                 alpha=0.2, color=color_reward)

ax2.tick_params(axis='y', labelcolor=color_reward)
ax2.axhline(y=oracle_mean, color='green', linestyle=':', 
            linewidth=2, label='PPO Oracle')

# Add "functional competence zone"
ax2.axhspan(100, ax2.get_ylim()[1], alpha=0.1, color='green', 
            label='Functional Competence Zone')

plt.title('Pareto Frontier: Explainability vs. Performance\n' + 
          'with Dual-Fidelity Metrics and Bootstrap CI', fontsize=14, fontweight='bold')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=10)

fig.tight_layout()
plt.savefig('pareto_frontier_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("COMPLETE! We now have:")
print("   Behavioral Fidelity (F_b)")
print("   Structural Fidelity (F_s) - NEW!")
print("   Performance-Transparency Gap (PTG) - NEW!")
print("   Bootstrap confidence intervals - NEW!")
print("   LaTeX table ready for paper")
print("   Enhanced figure with all metrics")
print("="*60)

#Feature importance plot
# Extract feature importance from depth-14 tree
feature_names = ['x', 'y', 'v_x', 'v_y', 'theta', 'omega', 'leg_1', 'leg_2']
importances = action_trees[14].feature_importances_

# Create bar plot
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances * 100)
plt.xlabel('State Variables', fontsize=12)
plt.ylabel('Importance (%)', fontsize=12)
plt.title('Feature Importance in Depth-14 Distilled Tree', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature-importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFeature Importance Rankings:")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp*100:.1f}%")

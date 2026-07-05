# Explainable Reinforcement Learning through Information-Theoretic Policy Compression

This repository contains the official code implementation for the paper **"Explainable Reinforcement Learning through Information-Theoretic Policy Compression"**.

The code provides a complete framework for distilling complex, opaque Deep Reinforcement Learning (DRL) models (e.g., Proximal Policy Optimization) into highly interpretable, capacity-constrained decision trees (CART) using a Value-Aware Information Bottleneck approach. It includes tools for evaluating these policies using our proposed Dual-Fidelity metrics (Structural Fidelity and Behavioral Fidelity) and calculating the Performance-Transparency Gap (PTG).

## Repository Structure

The repository is divided into four categories: the core experiments on the `LunarLander-v3` environment (including baseline comparisons), the cross-domain generalization experiments (MuJoCo, Blackjack, CartPole), and the figure-generation utility used to produce publication-quality plots.

### 1. Core Distillation & Baseline Comparisons (LunarLander-v3)
These scripts reproduce the foundational experiments, phase transition analysis, and baseline comparisons (Behavioral Cloning and VIPER) detailed in the paper.
* **`information_bottleneck_distillation.py`**: The primary script that trains the PPO Oracle on LunarLander-v3, extracts expert data alongside continuous value estimates, trains the distilled action and value trees, and outputs comprehensive evaluation metrics including the Pareto frontier and feature importance visualizations.
* **`baseline_comparison_experiments.py`**: Executes the comparative baselines (Behavioral Cloning and VIPER) on LunarLander-v3, computes Structural Fidelity (F_s) and Behavioral Fidelity (F_b), and generates the comprehensive multi-method comparison tables and plots.

### 2. Cross-Domain Generalization (MuJoCo, Blackjack, CartPole)
These scripts validate the value-aware distillation framework across highly diverse state-action spaces, bounding the Performance-Transparency Gap.

**MuJoCo HalfCheetah (continuous, high-dimensional):**
* **`generate_expert_dataset.py`**: Downloads a pre-trained PPO expert for MuJoCo HalfCheetah-v4 and generates rollout datasets containing states, actions, and value estimates $V(s)$.
* **`value_aware_distillation.py`**: Implements the joint-loss distillation algorithm (Cross-Entropy + Mean Squared Error) for continuous control spaces, mapping the expert's value landscape into the student tree.
* **`evaluate_distilled_policy.py`**: Evaluates the distilled continuous control policies over multiple episodes to determine empirical functional competence.

**Blackjack (discrete, stochastic):**
* **`generate_blackjack_dataset.py`**: Trains a PPO expert on the discrete, highly stochastic `Blackjack-v1` environment using a custom tuple-flattening observation wrapper and collects 50,000 expert rollouts.
* **`distill_blackjack_tree.py`**: Applies value-aware distillation to the discrete Blackjack dataset using one-hot encoded actions to jointly optimize classification accuracy and value regression within the decision tree.
* **`evaluate_blackjack_policy.py`**: Runs 100,000 evaluation episodes of the distilled Blackjack tree to rigorously calculate the true empirical win, loss, and draw rates against the dealer.

**CartPole (discrete, low-dimensional):**
* **`generate_cartpole_dataset.py`**: Trains a PPO expert on `CartPole-v1` (native flat observation, no wrapper required) and collects 50,000 expert rollouts.
* **`distill_cartpole_tree.py`**: Applies the same one-hot-encoded joint-tree value-aware distillation approach used for Blackjack to the CartPole dataset.
* **`evaluate_cartpole_policy.py`**: Runs 100 live evaluation episodes of the distilled CartPole tree and reports mean reward and the rate of episodes reaching the 500-step cap.

### 3. Figure Generation
* **`generate_figures.py`**: Regenerates the two publication figures — the main-manuscript Pareto Frontier plot and the Supplementary Material's Information Capacity plot — directly from the already-published summary statistics in Table I (fidelity, reward, and PTG values). Does not read any experiment output files; safe to re-run at any time to reproduce or restyle the figures without re-running any experiments.

## Pipeline Order

Each environment's experiments form an independent three-stage chain: **generate data → distill tree → evaluate tree**. No stage reads output from a different environment's chain. Run each chain top-to-bottom; skipping or reordering a stage within a chain will fail with a missing-file error.

```
LunarLander (self-contained, single script)
  information_bottleneck_distillation.py
      (trains PPO, distills all depths 2-14, evaluates, plots — all in one run)

  baseline_comparison_experiments.py
      (uses LunarLander's already-published Table I values as a fixed
       reference; runs its own fresh BC and VIPER baselines independently)

HalfCheetah
  generate_expert_dataset.py   -->  halfcheetah_expert_dataset.npz
      |
      v
  value_aware_distillation.py  -->  distilled_tree_depth_14.joblib
      |
      v
  evaluate_distilled_policy.py -->  console metrics only

Blackjack
  generate_blackjack_dataset.py -->  blackjack_expert_dataset.npz
      |
      v
  distill_blackjack_tree.py     -->  blackjack_tree_depth_7.joblib
      |
      v
  evaluate_blackjack_policy.py  -->  console metrics only

CartPole
  generate_cartpole_dataset.py -->  cartpole_expert_dataset.npz
      |
      v
  distill_cartpole_tree.py      -->  cartpole_tree_depth_4.joblib
      |
      v
  evaluate_cartpole_policy.py   -->  console metrics only

Figures (run any time, independent of all of the above)
  generate_figures.py -->  pareto-frontier.png/.pdf,
                            information_capacity_plot.png/.pdf
```

**Note:** `information_bottleneck_distillation.py` also produces its own internal Pareto frontier plot (`pareto_frontier_complete.png`) as part of its run. This has since been superseded by `generate_figures.py`, which reproduces the same figure with IEEE-style publication formatting — use `generate_figures.py`'s output for the manuscript, not the script's own inline plot.

## Program Reference Table

| Program | Environment | Input | Output | Needs T4 GPU? |
|---|---|---|---|---|
| `information_bottleneck_distillation.py` | LunarLander-v3 | None (trains from scratch) | `ppo_lunar_oracle.zip`, console metrics, LaTeX table, `pareto_frontier_complete.png`, `feature-importance.png` | No — MLP policy on non-image observations; CPU is sufficient |
| `baseline_comparison_experiments.py` | LunarLander-v3 | None (trains from scratch; references published Table I values internally) | Console metrics, comparison tables/plots | No — same as above |
| `generate_expert_dataset.py` | HalfCheetah-v4 | None (downloads pre-trained expert from Hugging Face) | `halfcheetah_expert_dataset.npz` | No — inference-only rollout collection, small MLP policy |
| `value_aware_distillation.py` | HalfCheetah-v4 | `halfcheetah_expert_dataset.npz` | `distilled_tree_depth_14.joblib` | No — scikit-learn `DecisionTreeRegressor`, no GPU code path exists |
| `evaluate_distilled_policy.py` | HalfCheetah-v4 | `distilled_tree_depth_14.joblib` | Console metrics only | No — sklearn inference + MuJoCo CPU stepping |
| `generate_blackjack_dataset.py` | Blackjack-v1 | None (trains from scratch) | `blackjack_expert_dataset.npz`, `ppo_blackjack_expert.zip` | No — MLP policy, tiny discrete state space |
| `distill_blackjack_tree.py` | Blackjack-v1 | `blackjack_expert_dataset.npz` | `blackjack_tree_depth_7.joblib` | No — scikit-learn only |
| `evaluate_blackjack_policy.py` | Blackjack-v1 | `blackjack_tree_depth_7.joblib` | Console metrics only | No — sklearn inference + environment stepping |
| `generate_cartpole_dataset.py` | CartPole-v1 | None (trains from scratch) | `cartpole_expert_dataset.npz`, `ppo_cartpole_expert.zip` | No — MLP policy, 4D observation space |
| `distill_cartpole_tree.py` | CartPole-v1 | `cartpole_expert_dataset.npz` | `cartpole_tree_depth_4.joblib` | No — scikit-learn only |
| `evaluate_cartpole_policy.py` | CartPole-v1 | `cartpole_tree_depth_4.joblib` | Console metrics only | No — sklearn inference + environment stepping |
| `generate_figures.py` | All (figure generation) | None (uses hardcoded, already-published Table I values) | `pareto-frontier.png/.pdf`, `information_capacity_plot.png/.pdf` | No — pure matplotlib rendering |

**On GPU usage generally:** none of the environments in this repository use image-based observations, so none require a convolutional policy network. Every PPO policy here is a small multilayer perceptron (SB3's default `MlpPolicy`), and every distillation step uses scikit-learn's CART implementation, which has no GPU code path at all. A T4 will not meaningfully speed up any script in this repository; a standard CPU runtime is sufficient throughout, and is recommended to conserve GPU quota for other work.

## Installation & Requirements

Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
pip install gymnasium[box2d,mujoco] stable-baselines3[extra] shimmy scikit-learn joblib pandas matplotlib seaborn huggingface_hub rl_zoo3
```
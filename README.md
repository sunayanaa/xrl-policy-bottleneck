# Explainable Reinforcement Learning through Information-Theoretic Policy Compression

This repository contains the official code implementation for the paper **"Explainable Reinforcement Learning through Information-Theoretic Policy Compression"**. 

The code provides a complete framework for distilling complex, opaque Deep Reinforcement Learning (DRL) models (e.g., Proximal Policy Optimization) into highly interpretable, capacity-constrained decision trees (CART) using a Value-Aware Information Bottleneck approach. It includes tools for evaluating these policies using our proposed Dual-Fidelity metrics (Structural Fidelity and Behavioral Fidelity) and calculating the Performance-Transparency Gap (PTG).

## Repository Structure

The repository is divided into two main categories: the core experiments on the `LunarLander-v3` environment (including baseline comparisons) and the cross-domain generalization experiments.

### 1. Core Distillation & Baseline Comparisons (LunarLander-v3)
These scripts reproduce the foundational experiments, phase transition analysis, and baseline comparisons (Behavioral Cloning and VIPER) detailed in the paper.
* **`information_bottleneck_distillation.py`**: The primary script that trains the PPO Oracle on LunarLander-v3, extracts expert data alongside continuous value estimates, trains the distilled action and value trees, and outputs comprehensive evaluation metrics including the Pareto frontier and feature importance visualizations.
* **`baseline_comparison_experiments.py`**: Executes the comparative baselines (Behavioral Cloning and VIPER) on LunarLander-v3, computes Structural Fidelity (F_s) and Behavioral Fidelity (F_b), and generates the comprehensive multi-method comparison tables and plots.

### 2. Cross-Domain Generalization (Blackjack & MuJoCo)
These scripts validate the value-aware distillation framework across highly diverse state-action spaces, bounding the Performance-Transparency Gap.
* **`generate_expert_dataset.py`**: Trains PPO experts for continuous control environments (e.g., MuJoCo HalfCheetah) and generates rollout datasets containing states, actions, and value estimates $V(s)$.
* **`value_aware_distillation.py`**: Implements the joint-loss distillation algorithm (Cross-Entropy + Mean Squared Error) for continuous control spaces, mapping the expert's value landscape into the student tree.
* **`evaluate_distilled_policy.py`**: Evaluates the distilled continuous control policies over multiple episodes to determine empirical functional competence.
* **`generate_blackjack_dataset.py`**: Trains a PPO expert on the discrete, highly stochastic `Blackjack-v1` environment using a custom tuple-flattening observation wrapper and collects 50,000 expert rollouts.
* **`distill_blackjack_tree.py`**: Applies value-aware distillation to the discrete Blackjack dataset using one-hot encoded actions to jointly optimize classification accuracy and value regression within the decision tree.
* **`evaluate_blackjack_policy.py`**: Runs 100,000 evaluation episodes of the distilled Blackjack tree to rigorously calculate the true empirical win, loss, and draw rates against the dealer.

## Installation & Requirements

Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
pip install gymnasium[box2d] stable-baselines3 shimmy scikit-learn pandas matplotlib seaborn
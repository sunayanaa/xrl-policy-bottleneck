# ==============================================================================
# Program Name: distill_cartpole_tree.py
# 
# Description: 
# Applies Value-Aware Policy Distillation to the discrete CartPole dataset.
# Uses one-hot encoding for actions to allow a single DecisionTreeRegressor to 
# jointly optimize discrete action classification and continuous value 
# regression via a concatenated multi-output target -- the same joint-tree 
# approach used for HalfCheetah and Blackjack (Option 1), rather than the 
# two-separate-trees approach used for LunarLander.
# ==============================================================================

# --- COLAB SETUP ---
!pip install -q scikit-learn joblib

import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from google.colab import drive
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Mount Google Drive
drive.mount('/content/drive')

def distill_discrete_tree(max_depth=4, value_weight=1.0):
    # 1. Setup paths
    drive_path = '/content/drive/My Drive/paper/XRL_Experiments/CartPole'
    dataset_file = os.path.join(drive_path, 'cartpole_expert_dataset.npz')
    model_save_file = os.path.join(drive_path, f'cartpole_tree_depth_{max_depth}.joblib')

    print("--- Starting Value-Aware Policy Distillation (CartPole) ---")

    # 2. Load the Dataset
    data = np.load(dataset_file)
    X_states = data['states']
    Y_actions_discrete = data['actions']
    Y_values = data['values'].reshape(-1, 1)

    print(f"Loaded {X_states.shape[0]} samples.")

    # 3. Formulate the Joint Objective for Discrete Actions
    # One-hot encode the actions (0=Push Left, 1=Push Right)
    num_classes = 2
    Y_actions_onehot = np.eye(num_classes)[Y_actions_discrete]

    # Scale values
    Y_values_scaled = Y_values * value_weight

    # Concatenate [Prob_Left, Prob_Right, Value]
    Y_joint = np.hstack((Y_actions_onehot, Y_values_scaled))

    # Split dataset
    X_train, X_test, Y_joint_train, Y_joint_test, Y_actions_train, Y_actions_test = train_test_split(
        X_states, Y_joint, Y_actions_discrete, test_size=0.2, random_state=42
    )

    # 4. Train the Information Bottleneck Tree
    print(f"\nTraining Capacity-Constrained CART Tree (Max Depth = {max_depth})...")
    # CartPole's 4-dimensional state space is low-dimensional; shallow depths
    # (e.g. 4-6) are expected to be sufficient -- try a couple of values to
    # find the shallowest depth that reliably solves the task
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X_train, Y_joint_train)
    print("Training complete!")

    # 5. Evaluate Fidelity on Test Set
    print("\n--- Evaluating Distillation Fidelity ---")
    predictions = tree.predict(X_test)

    # Extract predicted action probabilities and values
    pred_action_probs = predictions[:, :num_classes]
    pred_values = predictions[:, num_classes:] / value_weight
    true_values = Y_joint_test[:, num_classes:] / value_weight

    # The chosen action is the one with the highest predicted probability
    pred_actions_discrete = np.argmax(pred_action_probs, axis=1)

    # Calculate Behavioral Fidelity (Accuracy for discrete actions)
    behavioral_accuracy = accuracy_score(Y_actions_test, pred_actions_discrete)

    # Calculate Structural Fidelity (MSE for values)
    structural_mse = mean_squared_error(true_values, pred_values)

    print(f"Behavioral Fidelity (Action Accuracy): {behavioral_accuracy * 100:.2f}%")
    print(f"Structural Fidelity (Value MSE):       {structural_mse:.5f}")
    print(f"Tree Depth Achieved:                   {tree.get_depth()}")
    print(f"Total Leaf Nodes (Bottleneck Z):        {tree.get_n_leaves()}")

    # 6. Save the Distilled Model
    joblib.dump(tree, model_save_file)
    print(f"\nDistilled tree model saved to: {model_save_file}")

# Execute
# Try max_depth=4 first; if evaluation (evaluate_cartpole_policy.py) shows it
# doesn't reliably solve the task, retry with max_depth=6
distill_discrete_tree(max_depth=4, value_weight=1.5)

# ==============================================================================
# Program Name: distill_blackjack_tree.py
# 
# Description: 
# Applies Value-Aware Policy Distillation to the discrete Blackjack dataset.
# Uses one-hot encoding for actions to allow a DecisionTreeRegressor to 
# jointly optimize discrete action classification and continuous value regression.
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

def distill_discrete_tree(max_depth=7, value_weight=1.0):
    # 1. Setup paths
    drive_path = '/content/drive/My Drive/XRL_Experiments/Blackjack'
    dataset_file = os.path.join(drive_path, 'blackjack_expert_dataset.npz')
    model_save_file = os.path.join(drive_path, f'blackjack_tree_depth_{max_depth}.joblib')

    print("--- Starting Value-Aware Policy Distillation (Blackjack) ---")
    
    # 2. Load the Dataset
    data = np.load(dataset_file)
    X_states = data['states']
    Y_actions_discrete = data['actions']
    Y_values = data['values'].reshape(-1, 1)
    
    print(f"Loaded {X_states.shape[0]} samples.")

    # 3. Formulate the Joint Objective for Discrete Actions
    # One-hot encode the actions (0=Stick, 1=Hit)
    num_classes = 2
    Y_actions_onehot = np.eye(num_classes)[Y_actions_discrete]
    
    # Scale values
    Y_values_scaled = Y_values * value_weight
    
    # Concatenate [Prob_Stick, Prob_Hit, Value]
    Y_joint = np.hstack((Y_actions_onehot, Y_values_scaled))

    # Split dataset
    X_train, X_test, Y_joint_train, Y_joint_test, Y_actions_train, Y_actions_test = train_test_split(
        X_states, Y_joint, Y_actions_discrete, test_size=0.2, random_state=42
    )

    # 4. Train the Information Bottleneck Tree
    print(f"\nTraining Capacity-Constrained CART Tree (Max Depth = {max_depth})...")
    # For a simple game like Blackjack, depth 7 is usually more than enough
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
    print(f"Total Leaf Nodes (Bottleneck Z):       {tree.get_n_leaves()}")

    # 6. Save the Distilled Model
    joblib.dump(tree, model_save_file)
    print(f"\nDistilled tree model saved to: {model_save_file}")

# Execute
distill_discrete_tree(max_depth=7, value_weight=1.5)
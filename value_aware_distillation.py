# ==============================================================================
# Program Name: value_aware_distillation.py
# 
# Description: 
# This script applies the "Value-Aware Policy Distillation" algorithm to the 
# HalfCheetah expert dataset generated in the previous step.
# 
# It implements the joint-loss formulation described in Section IV.B of the 
# manuscript by concatenating the action and value targets, allowing standard 
# CART algorithms to perform value-aware state partitioning. 
# It strictly bounds the tree depth to enforce the Information Bottleneck (d <= 14).
# Finally, it evaluates both Behavioral Fidelity (Fb) and Structural Fidelity (Fs).
# ==============================================================================

# --- COLAB SETUP ---
!pip install -q scikit-learn joblib

import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from google.colab import drive
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Mount Google Drive to access the dataset and save the model
drive.mount('/content/drive')

def train_distilled_tree(max_depth=14, value_weight=1.0):
    """
    Loads expert data, trains a value-aware decision tree, and calculates fidelity.
    """
    # 1. Setup paths
    drive_path = '/content/drive/My Drive/paper/XRL_Experiments/MuJoCo'
    dataset_file = os.path.join(drive_path, 'halfcheetah_expert_dataset.npz')
    model_save_file = os.path.join(drive_path, f'distilled_tree_depth_{max_depth}.joblib')

    print("--- Starting Value-Aware Policy Distillation ---")
    
    # 2. Load the Dataset
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found at {dataset_file}. Please run dataset generation first.")
        
    print(f"Loading expert data from: {dataset_file}")
    data = np.load(dataset_file)
    X_states = data['states']
    Y_actions = data['actions']
    Y_values = data['values'].reshape(-1, 1) # Reshape to column vector
    
    print(f"Loaded {X_states.shape[0]} samples.")

    # 3. Formulate the Joint Objective (The Trick)
    # We scale the value function by the value_weight (lambda in the paper) 
    # to control how much the value landscape influences the tree splits.
    Y_values_scaled = Y_values * value_weight
    
    # Concatenate Actions and Scaled Values into a single Multi-Output target
    # Shape becomes (N, 7) -> 6 Action dimensions + 1 Value dimension
    Y_joint = np.hstack((Y_actions, Y_values_scaled))

    # Split into train and test sets to prove generalization
    X_train, X_test, Y_joint_train, Y_joint_test = train_test_split(
        X_states, Y_joint, test_size=0.2, random_state=42
    )

    # 4. Train the Information Bottleneck Tree
    print(f"\nTraining Capacity-Constrained CART Tree (Max Depth = {max_depth})...")
    # We use DecisionTreeRegressor because HalfCheetah actions are continuous torques
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X_train, Y_joint_train)
    
    print("Training complete!")

    # 5. Evaluate Fidelity on the Test Set
    print("\n--- Evaluating Distillation Fidelity ---")
    predictions = tree.predict(X_test)
    
    # Separate the predictions back into Actions and Values
    action_dims = Y_actions.shape[1]
    pred_actions = predictions[:, :action_dims]
    
    # Unscale the predicted values for accurate MSE evaluation
    pred_values = predictions[:, action_dims:] / value_weight
    true_values = Y_joint_test[:, action_dims:] / value_weight
    true_actions = Y_joint_test[:, :action_dims]

    # Calculate Behavioral Fidelity (Action MSE)
    behavioral_mse = mean_squared_error(true_actions, pred_actions)
    
    # Calculate Structural Fidelity (Value MSE)
    structural_mse = mean_squared_error(true_values, pred_values)

    print(f"Behavioral Fidelity (Action MSE): {behavioral_mse:.5f}")
    print(f"Structural Fidelity (Value MSE):  {structural_mse:.5f}")
    print(f"Tree Depth Achieved: {tree.get_depth()}")
    print(f"Total Leaf Nodes (Bottleneck Z): {tree.get_n_leaves()}")

    # 6. Save the Distilled Model
    joblib.dump(tree, model_save_file)
    print(f"\nDistilled tree model saved to: {model_save_file}")

# Execute the distillation
# You can experiment with different depths to show the Information Bottleneck effect
train_distilled_tree(max_depth=14, value_weight=2.0)
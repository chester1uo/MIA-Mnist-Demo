# Member Inference Attack (MIA) Example

This repository demonstrates a **Member Inference Attack** (MIA) using **PyTorch** and the **MNIST** dataset. MIA attempts to determine whether a specific sample was part of the training set of a (potentially remote) machine learning model. This codebase includes:

- **Target Model**: A neural network trained on a subset of the MNIST training data.  
- **Shadow Models**: Multiple models trained to mimic the target model’s behavior on different data subsets.  
- **Attack Model**: A classifier (enhanced MLP) that learns from the shadow models’ outputs to infer membership on the real target model.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Overview of the Process](#overview-of-the-process)  
5. [Key Components](#key-components)  
6. [Troubleshooting and Tips](#troubleshooting-and-tips)  
7. [References and Further Reading](#references-and-further-reading)

---

## Project Structure

The repository is organized into several Python files, each handling distinct parts of the MIA pipeline:

```
mia_example/
├── data.py            # Loads and splits MNIST data
├── model.py           # Defines neural network architectures (Target CNN, Enhanced AttackMLP)
├── train.py           # Training routines for target/shadow models and attack model
├── attack_dataset.py  # Creates the AttackDataset from shadow models' outputs
├── evaluation.py      # Evaluation routines for attack performance and final MIA
├── utils.py           # Utility functions: seed setting, checkpoint management, plotting
└── main.py            # Orchestrates the entire process: data loading, training, attacking
```

### Brief Descriptions

- **`data.py`**  
  Loads the MNIST dataset, splits it into a target training subset, multiple shadow subsets, and a global test set.

- **`model.py`**  
  Contains two classes:  
  1. `SimpleCNN` – A basic CNN for MNIST (used by both target and shadow models).  
  2. `EnhancedAttackMLP` – A multi-layer MLP with dropout, batch normalization, and advanced activations for performing membership inference.

- **`train.py`**  
  Defines functions to train any PyTorch model (`train_model`) and a specialized function (`train_attack_model`) for the attack model. Also includes optional features like checkpoint saving and early stopping.

- **`attack_dataset.py`**  
  Builds the **AttackDataset**, gathering outputs (logits, probabilities, entropy, correctness, etc.) from shadow models on both "member" (training data) and "non-member" (test data) samples.

- **`evaluation.py`**  
  Has functions to evaluate the attack model on the AttackDataset (`evaluate_attack`) and to perform the final member inference on the real target model (`member_inference`).

- **`utils.py`**  
  Helpers for setting random seeds, saving/loading checkpoints, plotting training loss, and plotting confusion matrices.

- **`main.py`**  
  The central script that runs all the steps:
  1. Data loading.  
  2. Training the target model.  
  3. Training multiple shadow models.  
  4. Creating the AttackDataset.  
  5. Training the attack model.  
  6. Evaluating the attack model’s performance and performing actual membership inference on the target model.

---

## Usage

1. **(Optional) Create a `checkpoints/` directory** for saving model checkpoints and plot images:
   ```bash
   mkdir checkpoints
   ```
2. **Run the main script**:
   ```bash
   python main.py
   ```
3. **Observe Outputs**:
   - Checkpoints for target, shadow, and attack models are saved in `checkpoints/`.  
   - Training loss plots and confusion matrix images are also saved in `checkpoints/`.  
   - Logs in the terminal or console will show training progress, metrics (loss, accuracy, F1), and final MIA performance (accuracy, ROC-AUC, etc.).

import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from data import load_data
from model import SimpleCNN, AttackMLP
from train import train_model, train_attack_model
from attack_dataset import AttackDataset
from evaluation import evaluate_attack, member_inference
from utils import set_seed, plot_loss, plot_confusion_matrix

def main():
    # 0. Set seed for reproducibility
    set_seed(42)

    # 1. Load data
    target_train_dataset, shadow_datasets, test_dataset, transform = load_data(seed=42, num_shadow_models=5)
    # Note: For demonstration, used num_shadow_models=2, but you can set it to 5 or more.

    # 2. Train (or Load) the target model
    print("[+] Training/Loading Target Model")
    target_model = SimpleCNN()
    target_train_loader = DataLoader(target_train_dataset, batch_size=64, shuffle=True)

    target_model, target_history = train_model(
        model=target_model,
        train_loader=target_train_loader,
        epochs=5,
        lr=0.001,
        save_path="checkpoints/target_model.pt",
        load_if_exists=True  # If True and checkpoint exists, it will load and skip training
    )
    # Plot target model training loss
    if target_history:
        plot_loss(target_history, title="Target Model Training Loss", save_path="checkpoints/target_loss.png")

    # 3. Train (or Load) shadow models
    print("[+] Training/Loading Shadow Models")
    shadow_models = []
    for i, shadow_dataset in enumerate(shadow_datasets):
        shadow_model = SimpleCNN()
        shadow_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)

        model_name   = f"checkpoints/shadow_model_{i}.pt"
        shadow_model, shadow_history = train_model(
            model=shadow_model,
            train_loader=shadow_loader,
            epochs=5,
            lr=0.001,
            save_path=model_name,
            load_if_exists=True
        )

        # Plot shadow training loss
        if shadow_history:
            plot_loss(shadow_history, title=f"Shadow Model {i} Training Loss", save_path=f"checkpoints/shadow_loss_{i}.png")

        shadow_models.append(shadow_model)

    # 4. Build the Attack Dataset from shadow models
    print("[+] Building Attack Dataset...")
    attack_dataset = AttackDataset(shadow_models, shadow_datasets, test_dataset)
    attack_loader = DataLoader(attack_dataset, batch_size=64, shuffle=True)

    # 5. Train (or Load) the Attack Model
    print("[+] Training/Loading Attack Model")
    attack_model = AttackMLP(input_size=10)
    attack_model, attack_history = train_attack_model(
        model=attack_model,
        attack_loader=attack_loader,
        epochs=100,
        lr=0.001,
        save_path="checkpoints/attack_model.pt",
        load_if_exists=False
    )
    # Plot attack training loss
    if attack_history:
        plot_loss(attack_history, title="Attack Model Training Loss", save_path="checkpoints/attack_loss.png")

    # 6. Evaluate the attack model (on the same AttackDataset)
    print("[+] Evaluating Attack Model on AttackDataset")
    labels, preds, cm = evaluate_attack(attack_model, attack_loader)
    plot_confusion_matrix(cm, title="Attack Model Confusion Matrix", save_path="checkpoints/cm_attack_model.png")

    # 7. Perform member inference on the actual target model
    print("[+] Performing Member Inference Attack on Target Model")
    target_labels, target_preds, target_cm = member_inference(
        target_model=target_model,
        attack_model=attack_model,
        target_dataset=target_train_dataset,
        test_dataset=test_dataset,
        num_samples=2000
    )
    plot_confusion_matrix(target_cm, title="Member Inference Confusion Matrix", save_path="checkpoints/cm_member_inference.png")

if __name__ == "__main__":
    main()

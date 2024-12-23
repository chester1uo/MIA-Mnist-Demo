import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    """Utility function to set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensures reproducible runs (though some deterministic operations might be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"[save_checkpoint] Model saved to {path}")

def load_checkpoint(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"[load_checkpoint] Model loaded from {path}")
    else:
        print(f"[load_checkpoint] No checkpoint found at {path}")

def plot_loss(history, title="Training Loss", save_path=None):
    epochs = history["epoch"]
    loss   = history["loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, marker='o', label='Loss')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[plot_loss] Figure saved to {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path:
        plt.savefig(save_path)
        print(f"[plot_confusion_matrix] Figure saved to {save_path}")
    else:
        plt.show()

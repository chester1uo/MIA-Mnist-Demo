import torch
from torch.utils.data import random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

def load_data(seed=42, num_shadow_models=5):
    """
    Loads MNIST data and splits it into:
      - target_train_dataset
      - shadow_datasets (for multiple shadow models)
      - test_dataset (for both shadow and final evaluation)
    """
    # Set manual seed for reproducibility
    g = torch.Generator().manual_seed(seed)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download MNIST
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = MNIST(root="./data", train=False, download=True, transform=transform)

    # Split train_dataset into target training set and shadow training set
    target_train_size = int(0.5 * len(train_dataset))
    shadow_train_size = len(train_dataset) - target_train_size

    target_train_dataset, shadow_train_dataset = random_split(
        train_dataset,
        [target_train_size, shadow_train_size],
        generator=g
    )

    # Further split shadow_train_dataset for multiple shadow models
    shadow_datasets = []
    shadow_size = shadow_train_size // num_shadow_models
    for i in range(num_shadow_models):
        start = i * shadow_size
        # For the last shadow model, take all remaining samples
        end = (i + 1) * shadow_size if i != (num_shadow_models - 1) else shadow_train_size
        indices = list(range(start, end))
        shadow_datasets.append(Subset(shadow_train_dataset, indices))

    return target_train_dataset, shadow_datasets, test_dataset, transform

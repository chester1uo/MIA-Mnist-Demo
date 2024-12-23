import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, train_loader, epochs=10, lr=0.001, save_path=None, load_if_exists=False):
    """
    Trains 'model' on train_loader and prints multiple metrics (loss, accuracy, F1) each epoch.
    
    Returns:
      - model
      - history (dict): stores loss and metrics for each epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # If requested, try to load checkpoint
    if load_if_exists and save_path is not None:
        import os
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print(f"[train_model] Loaded existing checkpoint from: {save_path}")
            return model, {}  # Return immediately if you don't want to retrain

    history = {
        "loss": [],
        "accuracy": [],
        "f1": [],
        "epoch": []
    }
    
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        # We'll gather all predictions and labels for computing metrics
        all_preds = []
        all_labels = []

        for data, target in train_loader:
            data   = data.to(device)
            target = target.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_loss += loss.item()

            # Predicted classes
            _, predicted = torch.max(outputs, dim=1)

            # Store predictions & labels for metrics
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(target.cpu().numpy())

        # Compute average loss per batch
        epoch_loss = running_loss / len(train_loader)

        # Flatten out all predictions/labels from the epoch
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        epoch_acc = accuracy_score(all_labels, all_preds)
        # "macro" F1 averages the F1 for each class equally;
        # you might use "micro" or "weighted" depending on your needs
        epoch_f1  = f1_score(all_labels, all_preds, average="macro")  

        # Store metrics in history
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)
        history["f1"].append(epoch_f1)
        history["epoch"].append(epoch)

        # Print progress
        print(f"Epoch [{epoch}/{epochs}] "
              f"-> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}% | F1: {epoch_f1:.3f}")
        
        # Save checkpoint if requested
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
            print(f"[train_model] Saved checkpoint to: {save_path}")
    return model, history

def train_attack_model(model, attack_loader, epochs=20, lr=0.001, save_path="./checkpoints", load_if_exists=False):
    """
    Trains the AttackMLP model.
    Returns:
      - model
      - history (dict): contains epoch-wise loss list
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load if checkpoint exists
    if load_if_exists and save_path is not None:
        import os
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print(f"[train_attack_model] Loaded existing checkpoint from: {save_path}")
            return model, {}

    history = {"loss": [], "epoch": []}

    print("[train_attack_model] Training start...")
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        model.train()
        for features, labels in attack_loader:
            features = features.to(device)
            labels   = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(attack_loader)
        history["loss"].append(epoch_loss)
        history["epoch"].append(epoch)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}")

    # Save checkpoint
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"[train_attack_model] Saved checkpoint to: {save_path}")

    return model, history

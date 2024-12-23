import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def evaluate_attack(model, attack_loader):
    """
    Evaluate the attack model on an AttackDataset-like loader.
    Prints accuracy, ROC-AUC, and also returns predictions & labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in attack_loader:
            features = features.to(device)
            labels   = labels.to(device)

            outputs = model(features).cpu().numpy()  # shape: (batch_size, 1)
            all_probs.extend(outputs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = (np.array(all_probs) > 0.5).astype(int)

    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc  = roc_auc_score(all_labels, all_probs)
    cm       = confusion_matrix(all_labels, all_preds)

    print(f"[Attack Model] Accuracy: {accuracy*100:.2f}% | ROC-AUC: {roc_auc:.2f}")

    return all_labels, all_preds, cm

def member_inference(target_model, attack_model, target_dataset, test_dataset, num_samples=1000):
    """
    Use the attack model to predict membership for actual target training samples
    and test (non-member) samples.
    Prints accuracy, ROC-AUC, returns predictions & labels, and confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model.to(device)
    attack_model.to(device)
    target_model.eval()
    attack_model.eval()

    import numpy as np
    from torch.utils.data import DataLoader, Subset

    # Pick random samples from target_dataset as "members"
    if len(target_dataset) > num_samples:
        member_indices = np.random.choice(len(target_dataset), size=num_samples, replace=False)
    else:
        member_indices = np.arange(len(target_dataset))
    member_subset = Subset(target_dataset, member_indices)

    # Collect member probabilities
    member_loader = DataLoader(member_subset, batch_size=64, shuffle=False)
    member_probs = []
    with torch.no_grad():
        for data, _ in member_loader:
            data = data.to(device)
            output = target_model(data)
            prob = torch.softmax(output, dim=1)
            member_probs.append(prob.cpu().numpy())
    member_probs = np.vstack(member_probs)
    member_labels = np.ones(len(member_probs))  # 1 => member

    # Pick random samples from test_dataset as "non-members"
    if len(test_dataset) > num_samples:
        non_member_indices = np.random.choice(len(test_dataset), size=num_samples, replace=False)
    else:
        non_member_indices = np.arange(len(test_dataset))
    non_member_subset = Subset(test_dataset, non_member_indices)

    # Collect non-member probabilities
    non_member_loader = DataLoader(non_member_subset, batch_size=64, shuffle=False)
    non_member_probs = []
    with torch.no_grad():
        for data, _ in non_member_loader:
            data = data.to(device)
            output = target_model(data)
            prob = torch.softmax(output, dim=1)
            non_member_probs.append(prob.cpu().numpy())
    non_member_probs = np.vstack(non_member_probs)
    non_member_labels = np.zeros(len(non_member_probs))  # 0 => non-member

    # Combine
    attack_features = np.vstack([member_probs, non_member_probs])
    attack_labels   = np.hstack([member_labels, non_member_labels])

    # Shuffle
    idx = np.random.permutation(len(attack_labels))
    attack_features = attack_features[idx]
    attack_labels   = attack_labels[idx]

    # Convert to tensors
    attack_features = torch.tensor(attack_features, dtype=torch.float32).to(device)
    attack_labels   = torch.tensor(attack_labels,   dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        outputs = attack_model(attack_features)
    all_probs = outputs.cpu().numpy().flatten()
    all_preds = (all_probs > 0.5).astype(int)
    all_labels= attack_labels.cpu().numpy().flatten()

    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc  = roc_auc_score(all_labels, all_probs)
    cm       = confusion_matrix(all_labels, all_preds)

    print(f"[Member Inference] Accuracy: {accuracy*100:.2f}% | ROC-AUC: {roc_auc:.2f}")

    return all_labels, all_preds, cm

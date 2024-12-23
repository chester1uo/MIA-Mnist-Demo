import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

class AttackDataset(Dataset):
    """
    Collects softmax probability vectors from shadow models
    for both 'member' (training data) and 'non-member' (test data).
    """

    def __init__(self, shadow_models, shadow_datasets, test_dataset):
        """
        Args:
            shadow_models: list of shadow model instances
            shadow_datasets: list of corresponding training Subsets for each shadow model
            test_dataset: dataset for 'non-member' samples
        """
        self.features = []
        self.labels   = []  # 1 for member, 0 for non-member
        self.shadow_models   = shadow_models
        self.shadow_datasets = shadow_datasets
        self.test_dataset    = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prepare_dataset()

    def get_probs(self, model, loader):
        model.eval()
        probs_list = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                output = model(data)
                prob = nn.functional.softmax(output, dim=1)
                probs_list.append(prob.cpu().numpy())
        return np.concatenate(probs_list, axis=0)
    
    def prepare_dataset(self):
        from torch.utils.data import DataLoader
        for shadow_model, shadow_dataset in zip(self.shadow_models, self.shadow_datasets):
            shadow_model.to(self.device)
            # Member
            shadow_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=False)
            shadow_member_probs = self.get_probs(shadow_model, shadow_loader)
            self.features.append(shadow_member_probs)
            self.labels += [1]*len(shadow_member_probs)

            # Non-member
            test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
            shadow_nonmember_probs = self.get_probs(shadow_model, test_loader)
            self.features.append(shadow_nonmember_probs)
            self.labels += [0]*len(shadow_nonmember_probs)

        self.features = np.vstack(self.features)
        self.labels   = np.array(self.labels)

        # Shuffle
        idx = np.random.permutation(len(self.labels))
        self.features = self.features[idx]
        self.labels   = self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],   dtype=torch.float32)
        )

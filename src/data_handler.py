from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class DataHandler(Dataset):
    def __init__(self, train_path, val_path, transform=None, batch_size=64, num_workers=2):
        self.train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
        self.val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        img, label = self.train_dataset[idx]
        return img, label
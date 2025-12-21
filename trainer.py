import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageNetFolderDataset(Dataset):
    def __init__(self, root_dir, labels_json, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load wnid -> class name
        with open(labels_json, "r") as f:
            self.wnid_to_name = json.load(f)

        # Create wnid -> integer label mapping
        self.wnids = sorted(self.wnid_to_name.keys())
        self.wnid_to_label = {wnid: i for i, wnid in enumerate(self.wnids)}

        self.samples = []

        for wnid in os.listdir(root_dir):
            wnid_path = os.path.join(root_dir, wnid)
            if not os.path.isdir(wnid_path):
                continue
            if wnid not in self.wnid_to_label:
                continue

            label = self.wnid_to_label[wnid]

            for fname in os.listdir(wnid_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(wnid_path, fname), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(
    batch_size=128,
    data_root="./imagenet",
    num_workers=4
):
    """
    ImageNet / ImageNet-100 DataLoader
    Replaces CIFAR-10 loader
    """

    train_dir = os.path.join(data_root, "train.X1")
    val_dir = os.path.join(data_root, "val.X")
    labels_json = os.path.join(data_root, "Labels.json")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageNetFolderDataset(
        root_dir=train_dir,
        labels_json=labels_json,
        transform=transform_train
    )

    val_dataset = ImageNetFolderDataset(
        root_dir=val_dir,
        labels_json=labels_json,
        transform=transform_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# def get_data_loaders(batch_size=128):
#     """Load CIFAR-10 dataset"""
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

#     train_dataset = datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=transform_train
#     )
#     test_dataset = datasets.CIFAR10(
#         root='./data', train=False, download=True, transform=transform_test
#     )

#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         num_workers=2, pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=2, pin_memory=True
#     )

#     return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss / total:.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


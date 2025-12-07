import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vit_small import *
from trainer import *


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create model
    config = TinyViTConfig()
    model = TinyViT(config).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params:,}')

    # Data loaders
    train_loader, test_loader = get_data_loaders(batch_size=128)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training loop
    num_epochs = 100
    best_acc = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'tiny_vit_best.pth')
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

    print(f'\nTraining complete! Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
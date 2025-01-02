import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataset import PneumoniaDataset
from model.model import resnet18
import torch
from torch import nn
import matplotlib
matplotlib.use('TkAgg')


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


def show_plot(num_epochs, train_losses, val_accuracies):
    # Plotting the loss and accuracy curves
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_losses, 'r', label='Training loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:green')
    ax2.plot(epochs, val_accuracies, 'g', label='Validation accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.title('Training and Validation Loss and Accuracy')
    plt.savefig('loss_accuracy_curve.png')
    plt.show()


def main(batch_size=32, num_epochs=20):
    data_dir = 'rsna-pneumonia-detection-challenge'
    pneumonia_dataset = PneumoniaDataset(data_dir, batch_size=batch_size)
    train_loader = pneumonia_dataset.train_loader
    val_loader = pneumonia_dataset.val_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18().to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        exp_lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        print()
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}%')

    show_plot(num_epochs, train_losses, val_accuracies)


if __name__ == '__main__':
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    main(batch_size=32, num_epochs=20)


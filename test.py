import torch
from tqdm import tqdm
from utils.dataset import PneumoniaDataset
from model.model import resnet18
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch import nn


def test_model(model_path, data_dir):
    pneumonia_dataset = PneumoniaDataset(data_dir)
    val_loader = pneumonia_dataset.val_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet18().to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, weights_only=True)) # Load the model weights
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')

    print(f'验证准确率: {accuracy * 100:.2f}%')
    print(f'精确度: {precision * 100:.2f}%')
    print(f'召回率: {recall * 100:.2f}%')
    print(f'F1 分数: {f1 * 100:.2f}%')


if __name__ == '__main__':
    model_path = 'best_model.pth'
    data_dir = 'rsna-pneumonia-detection-challenge'
    test_model(model_path, data_dir)

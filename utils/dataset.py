import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

class PneumoniaDataset:
    def __init__(self, data_dir, test_size=0.1, batch_size=32, transform=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform if transform else self.default_transform()
        self.label_data = pd.read_csv(os.path.join(data_dir, 'stage_2_train_labels.csv'))
        self.columns = ['patientId', 'Target']
        self.label_data = self.label_data.filter(self.columns)
        self.train_labels, self.val_labels = train_test_split(self.label_data.values, test_size=test_size)
        self.train_paths = [os.path.join(data_dir, 'stage_2_train_images', image[0]) for image in self.train_labels]
        self.val_paths = [os.path.join(data_dir, 'stage_2_train_images', image[0]) for image in self.val_labels]
        self.train_loader = self.create_data_loader(self.train_paths, self.train_labels)
        self.val_loader = self.create_data_loader(self.val_paths, self.val_labels, shuffle=False)

    def default_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor()
        ])

    def create_data_loader(self, paths, labels, shuffle=True):
        dataset = self.Dataset(paths, labels, transform=self.transform)
        return data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    class Dataset(data.Dataset):
        def __init__(self, paths, labels, transform=None):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __getitem__(self, index):
            image = dcmread(f'{self.paths[index]}.dcm')
            image = image.pixel_array
            image = image / 255.0
            image = (255 * image).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image).convert('RGB')
            label = self.labels[index][1]
            if self.transform is not None:
                image = self.transform(image)
            return image, label

        def __len__(self):
            return len(self.paths)

    def imshow(self, num_to_show=9):
        plt.figure(figsize=(10, 10))
        for i in range(num_to_show):
            plt.subplot(3, 3, i + 1)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            img_dcm = dcmread(f'{self.train_paths[i + 20]}.dcm')
            img_np = img_dcm.pixel_array
            plt.imshow(img_np, cmap=plt.cm.binary)
            plt.xlabel(self.train_labels[i + 20][1])
        plt.show()

    def show_batch(self):
        batch = iter(self.train_loader)
        images, labels = next(batch)
        image_grid = torchvision.utils.make_grid(images[:4])
        image_np = image_grid.numpy()
        img = np.transpose(image_np, (1, 2, 0))
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    data_dir = '../rsna-pneumonia-detection-challenge'
    pneumonia_dataset = PneumoniaDataset(data_dir)
    images, labels = next(iter(pneumonia_dataset.train_loader))
    print(images.shape)
    print(labels.shape)
    pneumonia_dataset.imshow()

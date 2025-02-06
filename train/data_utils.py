import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets
from PIL import Image

def load_synthetic(path):
    npzfile = np.load(path)
    images = npzfile['image']
    labels = npzfile['label']
    return images, labels

class SynData(Dataset):
    def __init__(self, path, transform=None):
        self.x, self.y = load_synthetic(path)
        self.transform=transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.x[idx], self.y[idx]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

class combine_dataloaders:
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
    
    def __iter__(self):
        return self._iterator()

    def __len__(self):
        return min(len(self.dataloader1), len(self.dataloader2))
    
    def _iterator(self):
        for (img1, label1), (img2, label2) in zip(self.dataloader1, self.dataloader2):
            images = torch.cat([img1, img2])
            labels = torch.cat([label1, label2])
            ids = torch.cat([torch.zeros(len(img1)), torch.ones(len(img2))])
            indices = torch.randperm(len(images))
            yield images[indices], labels[indices]#, ids[indices]

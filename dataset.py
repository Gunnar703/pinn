import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PINNDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_labels = [
            float(name.split(".")[0].replace("_", "."))
            for name in os.listdir(self.img_dir)
        ]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, "%s.dat" % str(self.img_labels[idx]).replace(".", "_")
        )
        image = torch.Tensor(np.loadtxt(img_path))
        label = self.img_labels[idx]
        return image, label

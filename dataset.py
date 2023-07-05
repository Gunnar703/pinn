import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PINNDataset(Dataset):
    def __init__(self, img_dir, device="cuda"):
        self.img_dir = img_dir
        self.img_labels = [
            float(name.split(".")[0].replace("_", "."))
            for name in os.listdir(self.img_dir)
        ]
        self.device = device

    def __len__(self):
        return 2
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, "%s.dat" % str(self.img_labels[idx]).replace(".", "_")
        )
        image = torch.Tensor(np.loadtxt(img_path)).permute(1, 0).to(self.device)
        label = self.img_labels[idx]
        return image, label


img_dir = os.path.join("data", "images")
dataset = PINNDataset(img_dir=img_dir)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

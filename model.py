import torch.nn as nn
from tqdm import tqdm
import torch


class CNNPINN(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.loss_weights = [1]
        self.losses = [nn.MSELoss()]
        self.callbacks = []
        self.optimizer = None

        self.layers = [
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=32, stride=2),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=16, stride=1),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=16, stride=1),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(40, 1),
        ]
        self.layers = nn.Sequential(*self.layers).to(self.device)

    def compile(self, optimizer, dataloader, **kwargs):
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.losses = kwargs["losses"] if "losses" in kwargs else self.losses
        self.loss_weights = (
            kwargs["loss_weights"] if "loss_weights" in kwargs else self.loss_weights
        )
        self.callbacks = (
            kwargs["callbacks"] if "callbacks" in kwargs else self.callbacks
        )
        self.default_loss = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

    def train(self, n_iterations: int = 100):
        for epoch in tqdm(range(n_iterations)):
            for img_batch, label_batch in self.dataloader:
                self.optimizer.zero_grad()
                pred_labels = self.forward(img_batch)
                weighted_losses = [
                    w
                    * l(
                        pred_labels.squeeze().to(self.device).float(),
                        label_batch.to(self.device).float(),
                    )
                    for w, l in zip(self.loss_weights, self.losses)
                ]
                loss = sum(weighted_losses)
                loss.backward()

                for callback in self.callbacks:
                    callback(
                        epoch=epoch,
                        weighted_losses=weighted_losses,
                        img_batch=img_batch,
                        label_batch=label_batch,
                    )

                self.optimizer.step()

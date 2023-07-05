import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import torch


class CNNPINN(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.loss_weights = [1]
        self.criterion = nn.MSELoss()
        self.losses = [
            # lambda img, pred, true: self.criterion(pred, true),
            lambda img, pred, true: self.phys_loss(img, pred),
        ]  # call signature: <function>(img, predicted_label, true_label)
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
        self.layers = torch.jit.script(self.layers)

        # Physics stuff
        self.t = torch.as_tensor(
            np.loadtxt(os.path.join("data", "t.txt"), max_rows=290)
        ).to(device)
        self.load = torch.as_tensor(
            np.loadtxt(os.path.join("data", "load.txt"), max_rows=290)
        ).to(device)
        self.load_mask = torch.zeros(4, 1).double().to(device)
        self.load_mask[3, 0] = 1

        self.M = torch.as_tensor(
            np.loadtxt(os.path.join("data", "M.txt")).reshape(4, 4)
        ).to(device)
        self.M_inv = torch.linalg.inv(self.M).to(device)
        self.k_basis = torch.as_tensor(
            np.loadtxt(os.path.join("data", "K_basis.txt"))
        ).to(device)
        self.a0, self.a1 = np.loadtxt(os.path.join("data", "Damp_params.txt"))

        self.compute_kc = lambda E: (
            self.k_basis * E,
            self.k_basis * E * self.a1 + self.M * self.a0,
        )

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
        return (
            self.layers(x) ** 2 * 1e8
        )  # squared to keep E positive, multiplied by 1e8 to keep E small

    def phys_loss(self, img, pred_label):
        u0 = torch.zeros(2, 4).to(self.device)

        def diff(u, t, E, load):
            pos = u[0, :].reshape(-1, 1).double().to(self.device)
            vel = u[1, :].reshape(-1, 1).double().to(self.device)

            K, C = self.compute_kc(E)
            acc = self.M_inv @ (load * self.load_mask - K @ pos - C @ vel)
            return torch.cat((vel, acc), dim=1).permute((1, 0))

        integrated_vel = torch.zeros((2, 290)).to(self.device)

        def rk_get_loss(u0, img, E):
            u = u0
            for idx in range(len(self.t) - 1):
                integrated_vel[0, idx] = u[1, 1].squeeze()
                integrated_vel[1, idx] = u[1, 3].squeeze()

                dt = self.t[idx + 1] - self.t[idx]
                t = self.t[idx]
                f = self.load[idx]

                k0 = diff(u, t, E, f)
                k1 = diff(u + k0 * dt / 2, t + dt / 2, E, f)
                k2 = diff(u + k1 * dt / 2, t + dt / 2, E, f)
                k3 = diff(u + k2 * dt, t + dt, E, f)

                du = 1 / 6 * (k0 + 2 * k1 + 2 * k2 + k3)
                u = u + du

            return integrated_vel

        predicted_images = []
        for cur_img, cur_E in zip(img, pred_label):
            predicted_images.append(rk_get_loss(u0, cur_img, cur_E).unsqueeze(0))
        pred_batch = torch.cat(predicted_images, dim=0).to(self.device)
        loss = self.criterion(pred_batch, img)
        return loss

    def train(self, n_iterations: int = 100):
        tr = tqdm(range(n_iterations))
        for epoch in tr:
            for img_batch, label_batch in self.dataloader:
                self.optimizer.zero_grad()
                pred_labels = self.forward(img_batch)
                weighted_losses = [
                    w
                    * l(
                        img_batch,
                        pred_labels.squeeze().to(self.device).float(),
                        label_batch.to(self.device).float(),
                    )
                    for w, l in zip(self.loss_weights, self.losses)
                ]
                loss = sum(weighted_losses)
                loss.backward()
                # tr.message = "loss = %.5g" % loss

                for callback in self.callbacks:
                    callback(
                        tr=tr,
                        epoch=epoch,
                        weighted_losses=weighted_losses,
                        img_batch=img_batch,
                        label_batch=label_batch,
                    )

                self.optimizer.step()

    def eval(self, eval_dataloader):
        preds = []
        labels = []
        for img_batch, label_batch in eval_dataloader:
            pred_labels = self.forward(img_batch)

            preds += list(pred_labels)
            labels += list(label_batch)

        percent_accuracy_total = 0
        for pred, label in zip(preds, labels):
            percent_error = torch.abs((pred - label) / label)
            percent_accuracy = 1 - percent_error

            percent_accuracy_total += percent_accuracy

        mapa = percent_accuracy / len(preds)
        return mapa

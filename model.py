import torch.nn as nn
import numpy as np
import torch
import os


class PINN(nn.Module):
    def __init__(
        self,
        layer_sizes: list[int],
        sigmas: list[float],
        dropout_rate: float = 0,
        device: str = "cuda",
    ):
        super().__init__()

        assert layer_sizes[1] % 2 == 0

        self.device = device
        self.activation = nn.Tanh()
        self.layer_sizes = layer_sizes
        self.sigmas = sigmas
        self.dropout_rate = dropout_rate
        self.callbacks = []
        self.loss_weights = [1, 1, 1]

        self.criterion = nn.MSELoss()

        self.optimizer = None

        self.a = torch.rand(1).to(device).requires_grad_(True)

        self.b = []
        for sigma in self.sigmas:
            self.b.append(
                torch.normal(
                    mean=0, std=sigma, size=(layer_sizes[0], layer_sizes[1] // 2)
                ).to(device)
            )

        self.linears = []
        for i in range(2, len(layer_sizes) - 1):
            self.linears.append(
                nn.Linear(layer_sizes[i - 1], layer_sizes[i]).to(device)
            )

        self._dense = nn.Linear(layer_sizes[-2] * len(sigmas), layer_sizes[-1]).to(
            device
        )

    def E(self):
        return self.a**2

    def forward(self, inputs):
        x = inputs

        # fourier feature layer
        yb = [
            self._fourier_feature_forward(x, self.b[i]) for i in range(len(self.sigmas))
        ]
        y = [elem[0] for elem in yb]
        self.fourier_feature_weights = [elem[1] for elem in yb]

        # fully-connected layers
        y = [self._fully_connected_forward(_y) for _y in y]

        # concatenate all the fourier features
        y = torch.cat(y, axis=1)
        y = self._dense(y)
        return y

    def _fourier_feature_forward(self, y, b):
        y = torch.cat(
            [torch.cos(torch.matmul(y, b)), torch.sin(torch.matmul(y, b))], dim=1
        )
        return y, b

    def _fully_connected_forward(self, y):
        for linear in self.linears:
            y = self.activation(linear(y))
            if self.dropout_rate > 0:
                y = torch.nn.functional.dropout(
                    y, p=self.dropout_rate, training=self.training
                ).to(self.device)
        return y

    def predict(self, x):
        x = torch.Tensor(x).to(self.device)
        self.training = False
        return self.forward(x).numpy()

    def compile(self, optimizer, **kwargs):
        self.optimizer = optimizer
        if "callbacks" in kwargs:
            self.callbacks = kwargs["callbacks"]
        if "loss_weights" in kwargs:
            self.loss_weights = kwargs["loss_weights"]

    def load_ops_data(self, data_file: str = "data", max_rows: int = 290):
        self.M = np.loadtxt(os.path.join(data_file, "M.txt"))
        self.C = np.loadtxt(os.path.join(data_file, "C.txt"))
        self.K = np.loadtxt(os.path.join(data_file, "K.txt"))
        self.Y = np.loadtxt(os.path.join(data_file, "Y.txt"))
        self.time = np.loadtxt(os.path.join(data_file, "t.txt"), max_rows=max_rows)
        self.load = np.loadtxt(os.path.join(data_file, "load.txt"), max_rows=max_rows)
        self.K_basis = np.loadtxt(os.path.join(data_file, "k_basis.txt"))
        self.a0, self.a1 = np.loadtxt(os.path.join(data_file, "Damp_param.txt"))

        self.node3_disp_y = np.loadtxt(
            os.path.join(data_file, "Disp_3_2D.txt"), max_rows=max_rows
        )
        self.node4_disp_y = np.loadtxt(
            os.path.join(data_file, "Disp_4_2D.txt"), max_rows=max_rows
        )

        self.node3_vel_y = np.loadtxt(
            os.path.join(data_file, "Vel_3_2D.txt"), max_rows=max_rows
        )
        self.node4_vel_y = np.loadtxt(
            os.path.join(data_file, "Vel_4_2D.txt"), max_rows=max_rows
        )

        ## Normalize
        self.T_MAX = self.time.max()
        self.U_MAX = max(self.node3_disp_y.max(), self.node4_disp_y.max())

        self.time /= self.T_MAX
        self.node3_vel_y *= self.T_MAX / self.U_MAX
        self.node4_vel_y *= self.T_MAX / self.U_MAX

        ## Convert to tensors
        (
            self.M,
            self.C,
            self.K,
            self.time,
            self.load,
            self.K_basis,
            self.node3_vel_y,
            self.node4_vel_y,
            self.node3_disp_y,
            self.node4_disp_y,
        ) = (
            torch.Tensor(self.M).to(self.device),
            torch.Tensor(self.C).to(self.device),
            torch.Tensor(self.K).to(self.device),
            torch.Tensor(self.time).to(self.device),
            torch.Tensor(self.load).to(self.device),
            torch.Tensor(self.K_basis).to(self.device),
            torch.Tensor(self.node3_vel_y).to(self.device),
            torch.Tensor(self.node4_vel_y).to(self.device),
            torch.Tensor(self.node3_disp_y).to(self.device),
            torch.Tensor(self.node4_disp_y).to(self.device),
        )

    def physics_loss(self, x, y):
        M = self.M
        K = self.E() * self.K_basis * 1e8
        C = self.a0 * M + self.a1 * K

        F = y * 0
        F[:, 3] = torch.Tensor(
            np.interp(
                x.detach().cpu().numpy(),
                self.time.cpu().numpy(),
                self.load.cpu().numpy(),
            )
            * -1e3
        ).squeeze()
        F = F.permute((1, 0))

        y_t, y_tt = [y * 0] * 2
        for dim in range(y.shape[1]):
            # u for all times at the current node
            u_current = y[:, dim].unsqueeze(1).to(self.device)

            # du/dt for all times at the current node
            du_dt_current = torch.autograd.grad(
                u_current,
                x,
                grad_outputs=torch.ones_like(u_current),
                retain_graph=True,
                create_graph=True,
            )[0].to(self.device)

            # d^2u/dt^2 for all times at the current node
            d2u_dt2_current = torch.autograd.grad(
                du_dt_current,
                x,
                grad_outputs=torch.ones_like(du_dt_current),
                retain_graph=True,
                create_graph=True,
            )[0].to(self.device)

            y_t[:, dim] = du_dt_current.squeeze()
            y_tt[:, dim] = d2u_dt2_current.squeeze()

        Y, Y_T, Y_TT = y.permute((1, 0)), y_t.permute((1, 0)), y_tt.permute((1, 0))

        residual = M @ Y_TT + C @ Y_T + K @ Y - F
        return self.criterion(residual, residual * 0)

    def data_loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true)

    def sample_physics_points(self, N_REGIONS, N_POINTS):
        length = 1 / N_REGIONS
        points_per_region = int(N_POINTS / N_REGIONS)

        xlist = []
        for i in range(N_REGIONS):
            val = torch.rand(points_per_region)
            val *= length
            val += length * i
            xlist.append(val)
        xlist = torch.cat(xlist).to(self.device)
        return xlist

    def train(self, iterations: int = 50_000):
        for epoch in range(iterations):
            self.optimizer.zero_grad()

            data_t = self.time.view(-1, 1).requires_grad_(True)
            u_pred = self.forward(data_t)

            if epoch % 4000 == 0:
                phys_t = (
                    self.sample_physics_points(N_REGIONS=2, N_POINTS=1000)
                    .view(-1, 1)
                    .requires_grad_(True)
                )
            u_pred_phys = self.forward(phys_t)

            u_pred_t = u_pred * 0

            for dim in range(u_pred.shape[1]):
                # u for all times at the current node
                u_current_data = u_pred[:, dim].unsqueeze(1).to(self.device)

                # du/dt for all times at the current node
                du_dt_current_data = torch.autograd.grad(
                    u_current_data,
                    data_t,
                    grad_outputs=torch.ones_like(u_current_data),
                    retain_graph=True,
                    create_graph=True,
                )[0].to(self.device)

                u_pred_t[:, dim] = du_dt_current_data.squeeze()

            physics_loss = self.physics_loss(phys_t, u_pred_phys)
            data_loss1 = self.data_loss(u_pred_t[:, 1].squeeze(), self.node3_vel_y)
            data_loss2 = self.data_loss(u_pred_t[:, 3].squeeze(), self.node4_vel_y)

            loss = (
                self.loss_weights[0] * physics_loss
                + self.loss_weights[1] * data_loss1
                + self.loss_weights[2] * data_loss2
            )
            loss.backward()

            for callback in self.callbacks:
                callback(
                    model=self,
                    epoch=epoch,
                    physics_loss=physics_loss * self.loss_weights[0],
                    data_loss=[
                        data_loss1 * self.loss_weights[1],
                        data_loss2 * self.loss_weights[2],
                    ],
                    u_pred_t=u_pred_t,
                    u_pred=u_pred,
                    data_t=data_t,
                )

            self.optimizer.step()

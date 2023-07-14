from copy import deepcopy
import numpy as np
import torch


class PINN(torch.nn.Module):
    def __init__(self, layers, activation, data, **kw):
        super().__init__()

        assert layers[0] == layers[-1] == 8

        linears = [
            torch.nn.Linear(in_features=layers[i - 1], out_features=layers[i])
            for i in range(1, len(layers) - 1)
        ]
        layers = torch.nn.Sequential(
            *[torch.nn.Sequential(linear, activation) for linear in linears],
            torch.nn.Linear(in_features=layers[-2], out_features=layers[-1])
        )
        self.layers = torch.jit.script(layers)

        # Data
        self.data = data
        self.data_tensor = deepcopy(data)
        self.data_tensor.torchify()

        self.t, self.load, self.u1, self.u3 = (
            (
                self.data.t,
                self.data.load,
                self.data.pos[1, :],
                self.data.pos[3, :],
            )
            if not all([x in kw for x in ("t", "load", "u1", "u3")])
            else (kw["t"], kw["load"], kw["u1"], kw["u3"])
        )

        # Parameters
        self.loss_weight = torch.rand(1).requires_grad_(True)
        self.E = torch.rand(1).requires_grad_(True)
        self.extra_params = [self.E, self.loss_weight]

    def forward(self):
        """Perform a forward pass of the model over self.t

        Returns:
            torch.Tensor: model evaluated at each time-step in t
        """
        y_history = torch.zeros((8, len(self.t)))

        # Start with u(0) = u'(x) = 0
        y = torch.zeros(8, 1)

        for idx in range(self.t.shape[0]):
            y_history[:, idx] = y.squeeze()
            y = self.layers(y).reshape(-1, 1)

        return y_history

    def compute_physics_loss(self, y_history):
        """
        Compute the error due to

        M u_tt + C u_t + K u = f

        where y_history = [
            [u0(self.t0),   u0(self.t1),   ..., u0(self.tn)  ]
                                    ...
            [u3(self.t0),   u3(self.t1),   ..., u3(self.tn)  ]
            [u1_t(self.t0), u1_t(self.t1), ..., u1_t(self.tn)]
                                    ...
            [u4_t(self.t0), u4_t(self.t1), ..., u4_t(self.tn)]
        ]
        """
        u = y_history[]

    def train(self, optimizer, lr_scheduler=None, callbacks=[], epochs=1000):
        self.epoch = 0

        def closure():
            optimizer.zero_grad()
            epoch_summary = {}

            y_history = self.forward()
            physics_loss = self.compute_physics_loss(y_history)
            data_loss = self.compute_data_loss(y_history)

            loss = (1 - self.loss_weight) * physics_loss + (
                self.loss_weight
            ) * data_loss

            epoch_summary = {
                "epoch": self.epoch,
                "loss": loss.detach().cpu(),
                "phys_loss": physics_loss.detach().cpu(),
                "data_loss": data_loss.detach().cpu(),
                "loss_weight": self.loss_weight.detach().cpu(),
                "E": self.E.detach().cpu(),
            }

            loss.backward()

            for callback in callbacks:
                callback(**epoch_summary)

            self.epoch += 1
            return loss

        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(closure)
            return

        for epoch in epochs:
            self.epoch = epoch
            loss = closure()
            loss.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

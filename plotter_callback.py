import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
import torch


class PlotterCallback(dde.callbacks.Callback):
    def __init__(
        self, period, filepath, data, tsol, usol, E_learned, plot_residual=False
    ):
        super().__init__()
        self.period = period
        self.filepath = filepath
        self.epoch = -1
        self.data = data
        self.tsol = tsol
        self.usol = usol
        self.E_learned = E_learned
        self.plot_residual = plot_residual

    def on_epoch_end(self):
        self.epoch += 1  # increment epoch counter
        if self.epoch % self.period != 0:
            return
        fig, axes = plt.subplots(4, 1, figsize=(8, 6))
        fig.suptitle(
            f"Epoch: {self.epoch}"
            + "\n"
            + "E = %.3f" % float(self.E_learned.detach().cpu().item())
            + "\n"
        )

        residual = self.model.predict(
            self.data["t"].reshape(-1, 1), operator=self.get_residual
        )
        v_pred = self.model.predict(
            self.data["t"].reshape(-1, 1), operator=self.differentiate_model_output
        )
        for dim in range(4):
            ax = axes[dim]

            ax.plot(self.data["t"], v_pred[:, dim], label="Prediction", color="black")
            ax.plot(
                self.data["t"],
                residual[:, dim] * 1e-5,
                label=r"Residual \times 10^{-5}",
                color="purple",
                linestyle="--",
            )

            # Plot Solution Data
            ax.plot(
                self.tsol,
                self.usol[dim],
                label="Solution (RK-45)",
                color="gray",
            )

            # Plot given data
            if dim == 0:
                ax.plot(
                    self.data["t"],
                    self.data["Vel_3_1_2D"],
                    label="Data (OPS)",
                    marker="x",
                    markersize=1,
                    linestyle="None",
                    color="orange",
                )
            elif dim == 1:
                ax.plot(
                    self.data["t"],
                    self.data["Vel_3_2D"],
                    label="Data (OPS)",
                    marker="x",
                    markersize=1,
                    linestyle="None",
                    color="orange",
                )
            elif dim == 2:
                ax.plot(
                    self.data["t"],
                    self.data["Vel_4_1_2D"],
                    label="Data (OPS)",
                    marker="x",
                    markersize=1,
                    linestyle="None",
                    color="orange",
                )
            elif dim == 3:
                ax.plot(
                    self.data["t"],
                    self.data["Vel_4_2D"],
                    label="Data (OPS)",
                    marker="x",
                    markersize=1,
                    linestyle="None",
                    color="orange",
                )

            ax.set_ylabel(r"$\dot{u}_%s(t)$" % (dim))
            if dim == 3:
                ax.set_xlabel(r"Time ($t$)")
            if dim == 0:
                ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
        plt.savefig(
            f"{self.filepath}/epoch_{self.epoch}_prediction.png", bbox_inches="tight"
        )
        plt.close()

        # This shouldn't stay here, but for now it's convenient
        if self.epoch == 200_000:
            self.model.losshistory.set_loss_weights([1, 1, 1, 1, 1])

    def differentiate_model_output(self, x, y):
        ret = torch.zeros_like(y)
        for i in range(4):
            ret[:, i] = dde.grad.jacobian(y, x, i=i, j=0).squeeze()
        return ret

    def get_residual(self, t, u):
        y = u
        y_t = torch.zeros_like(y)
        y_tt = torch.zeros_like(y)

        for dim in range(4):
            y_t[:, dim] = dde.grad.jacobian(u, t, i=dim, j=0).squeeze()
            y_tt[:, dim] = dde.grad.hessian(u, t, component=dim).squeeze()

        E = torch.abs(self.E_learned) * 1e7
        K = torch.Tensor(self.data["k_basis"]) * E
        C = (
            self.data["Damp_param"][0] * torch.Tensor(self.data["M"])
            + self.data["Damp_param"][1] * K
        )

        F = np.zeros((t.shape[0], u.shape[1]))
        f_quasiscalar = self.force_magnitude(t.detach().cpu()).squeeze()
        F[:, 3] = -f_quasiscalar
        F = torch.Tensor(F)

        residual = (
            torch.mm(torch.Tensor(self.data["M"]), y_tt.permute((1, 0)))
            + torch.mm(torch.abs(C), y_t.permute((1, 0)))
            + torch.mm(torch.abs(K), y.permute((1, 0)))
            - F.permute((1, 0))
        ).permute((1, 0))

        return residual

    def force_magnitude(self, t):
        return np.interp(t, self.data["t"], self.data["load"]) * 1e3

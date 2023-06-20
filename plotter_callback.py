import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate
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

        self.k_basis = torch.Tensor(self.data["k_basis"]).to("cuda")
        self.M = torch.Tensor(self.data["M"]).to("cuda")

        t = data["t"]
        self.du_t_1 = data["Vel_3_2D"]
        self.du_t_3 = data["Vel_4_2D"]

        self.u_1 = integrate.cumulative_trapezoid(self.du_t_1, t, initial=0)
        self.u_3 = integrate.cumulative_trapezoid(self.du_t_3, t, initial=0)

        self.du_tt_1, self.du_tt_3 = [np.zeros_like(self.du_t_1)] * 2
        self.du_tt_1[1:] = (self.du_t_1[1:] - self.du_t_1[:-1]) / (t[1:] - t[:-1])
        self.du_tt_3[1:] = (self.du_t_3[1:] - self.du_t_3[:-1]) / (t[1:] - t[:-1])
        self.t = t

        (
            self.du_t_1_np,
            self.du_t_3_np,
            self.u_1_np,
            self.u_3_np,
            self.du_tt_1_np,
            self.du_tt_3_np,
        ) = (
            self.du_t_1.copy(),
            self.du_t_3.copy(),
            self.u_1.copy(),
            self.u_3.copy(),
            self.du_tt_1.copy(),
            self.du_tt_3.copy(),
        )

        (
            self.du_t_1,
            self.du_t_3,
            self.u_1,
            self.u_3,
            self.du_tt_1,
            self.du_tt_3,
        ) = (
            torch.Tensor(self.du_t_1).to("cuda"),
            torch.Tensor(self.du_t_3).to("cuda"),
            torch.Tensor(self.u_1).to("cuda"),
            torch.Tensor(self.u_3).to("cuda"),
            torch.Tensor(self.du_tt_1).to("cuda"),
            torch.Tensor(self.du_tt_3).to("cuda"),
        )

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

            if dim == 0:
                ax.plot(self.t, v_pred[:, 0], label="Prediction", color="black")
                ax.plot(
                    self.data["t"],
                    residual[:, 0] * 1e-6,
                    label=r"Residual $\times 10^{-6}$",
                    color="purple",
                    linestyle="--",
                )
            elif dim == 1:
                ax.plot(self.t, self.du_t_1_np, label="Hard BC", color="gray")
                ax.plot(self.t, self.data["Vel_3_1_2D"], label="Solution", color="gray")
            elif dim == 2:
                ax.plot(self.t, v_pred[:, 1], label="Prediction", color="black")
                ax.plot(self.t, self.data["Vel_4_1_2D"], label="Solution", color="gray")
                ax.plot(
                    self.data["t"],
                    residual[:, 1] * 1e-6,
                    label=r"Residual $\times 10^{-6}$",
                    color="purple",
                    linestyle="--",
                )
            elif dim == 3:
                ax.plot(self.t, self.du_t_3_np, label="Hard BC", color="gray")

            ax.set_ylabel(r"$\dot{u}_%s(t)$" % (dim))
            if dim == 3:
                ax.set_xlabel(r"Time ($t$)")
            if dim == 0:
                ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.25))
        plt.savefig(
            f"{self.filepath}/epoch_{self.epoch}_prediction.png", bbox_inches="tight"
        )
        plt.close()

    def differentiate_model_output(self, x, y):
        ret = torch.zeros_like(y)
        for i in range(2):
            ret[:, i] = dde.grad.jacobian(y, x, i=i, j=0).squeeze()
        return ret

    def get_residual(self, t, u):
        y = u
        y_t = torch.zeros_like(y)
        y_tt = torch.zeros_like(y)

        for dim in (0, 1):
            y_t[:, dim] = dde.grad.jacobian(u, t, i=dim, j=0).squeeze()
            y_tt[:, dim] = dde.grad.hessian(u, t, component=dim).squeeze()

        y = torch.concatenate(
            (
                y[:, 0].reshape(-1, 1),
                self.u_1.reshape(-1, 1),
                y[:, 1].reshape(-1, 1),
                self.u_3.reshape(-1, 1),
            ),
            1,
        )

        y_t = torch.concatenate(
            (
                y_t[:, 0].reshape(-1, 1),
                self.du_t_1.reshape(-1, 1),
                y_t[:, 1].reshape(-1, 1),
                self.du_t_3.reshape(-1, 1),
            ),
            1,
        )

        y_tt = torch.concatenate(
            (
                y_tt[:, 0].reshape(-1, 1),
                self.du_tt_1.reshape(-1, 1),
                y_tt[:, 1].reshape(-1, 1),
                self.du_tt_3.reshape(-1, 1),
            ),
            1,
        )

        E = torch.abs(self.E_learned) * 1e7
        K = self.k_basis * E
        C = self.data["Damp_param"][0] * self.M + self.data["Damp_param"][1] * K

        F = np.zeros((t.shape[0], 4))
        f_quasiscalar = self.force_magnitude(t.detach().cpu()).squeeze()
        F[:, 3] = -f_quasiscalar
        F = torch.Tensor(F)

        residual = (
            torch.mm(self.M, y_tt.permute((1, 0)))
            + torch.mm(torch.abs(C), y_t.permute((1, 0)))
            + torch.mm(torch.abs(K), y.permute((1, 0)))
            - F.permute((1, 0))
        ).permute((1, 0))

        return residual

    def force_magnitude(self, t):
        return np.interp(t, self.data["t"], self.data["load"]) * 1e3

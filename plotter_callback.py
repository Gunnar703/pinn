import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate
import torch


class PlotterCallback(dde.callbacks.Callback):
    def __init__(
        self, period, filepath, data, E_learned, t_max, u_max, plot_residual=False
    ):
        super().__init__()
        self.period = period
        self.filepath = filepath
        self.epoch = -1
        self.data = data
        self.plot_residual = plot_residual
        self.T_MAX = t_max
        self.U_MAX = u_max

        self.E_learned = E_learned

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

    def on_epoch_begin(self):
        self.epoch += 1  # increment epoch counter
        if self.epoch % self.period != 0:
            return
        fig, axes = plt.subplots(4, 1, figsize=(8, 6))
        fig.suptitle(f"Epoch: {self.epoch}" + "\n" + "E = %.3g" % self.E_learned**2)

        v_pred = self.model.predict(
            self.data["t"].reshape(-1, 1), operator=self.differentiate_model_output
        )
        for dim in range(4):
            ax = axes[dim]
            ax.plot(self.t, v_pred[:, dim], "--", label="Prediction", color="purple")

            if dim == 0:
                ax.plot(self.t, self.data["Vel_3_1_2D"], label="Solution", color="gray")
            elif dim == 1:
                ax.plot(self.t, self.data["Vel_3_2D"], label="Solution", color="gray")
                ax.plot(
                    self.t,
                    self.data["Vel_3_2D"],
                    linestyle="None",
                    marker="+",
                    markersize=2,
                    label="Data",
                    color="orange",
                )
            elif dim == 2:
                ax.plot(self.t, self.data["Vel_4_1_2D"], label="Solution", color="gray")
            elif dim == 3:
                ax.plot(self.t, self.data["Vel_4_2D"], label="Solution", color="gray")
                ax.plot(
                    self.t,
                    self.data["Vel_4_2D"],
                    linestyle="None",
                    marker="+",
                    markersize=2,
                    label="Data",
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

    def differentiate_model_output(self, x, y):
        ret = torch.zeros_like(y)
        for i in range(y.shape[1]):
            ret[:, i] = dde.grad.jacobian(y, x, i=i, j=0).squeeze()
        return ret

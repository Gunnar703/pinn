import deepxde as dde
from matplotlib import pyplot as plt
import torch


class PlotterCallback(dde.callbacks.Callback):
    def __init__(self, period, filepath, data, tsol, usol_derivative):
        super().__init__()
        self.period = period
        self.filepath = filepath
        self.epoch = -1
        self.data = data
        self.tsol = tsol
        self.usol_derivative = usol_derivative

    def on_epoch_end(self):
        self.epoch += 1  # increment epoch counter
        if self.epoch % self.period != 0:
            return
        fig, axes = plt.subplots(4, 1, figsize=(8, 6))
        fig.suptitle(f"Epoch: {self.epoch}" + "\n")

        v_pred = self.model.predict(
            self.data["t"].reshape(-1, 1), operator=self.differentiate_model_output
        )
        for dim in range(4):
            ax = axes[dim]

            ax.plot(self.data["t"], v_pred[:, dim], label="Prediction", color="black")

            # Plot Solution Data
            ax.plot(
                self.tsol,
                self.usol_derivative[dim],
                label="Solution (RK-45)",
                color="gray",
            )

            # Plot given data
            if dim == 1:
                ax.plot(
                    self.data["t"],
                    self.data["Vel_3_2D"],
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
            else:
                ax.plot(
                    0,
                    0,
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
                ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
        plt.savefig(
            f"{self.filepath}/epoch_{self.epoch}_prediction.png", bbox_inches="tight"
        )
        plt.close()

    def differentiate_model_output(self, x, y):
        ret = torch.zeros_like(y)
        for i in range(4):
            ret[:, i] = dde.grad.jacobian(y, x, i=i, j=0).squeeze()
        return ret

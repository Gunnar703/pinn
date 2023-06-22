import deepxde as dde
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate
import torch


class PlotterCallback(dde.callbacks.Callback):
    def __init__(self, period, filepath, data, E, t_max, u_max, plot_residual=False):
        super().__init__()
        self.period = period
        self.filepath = filepath
        self.epoch = -1
        self.data = data
        self.E = E
        self.plot_residual = plot_residual
        self.T_MAX = t_max
        self.U_MAX = u_max

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
            + "E = %.3f" % float(self.E.detach().cpu().item())
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
                # ax.plot(
                #     self.data["t"],
                #     residual[:, 0] * 1e-6,
                #     label=r"Residual $\times 10^{-6}$",
                #     color="purple",
                #     linestyle="--",
                # )
                ax.plot(self.t, self.data["Vel_3_1_2D"], label="Solution", color="gray")
            elif dim == 1:
                ax.plot(self.t, self.du_t_1_np, label="Hard BC", color="gray")
            elif dim == 2:
                ax.plot(self.t, v_pred[:, 1], label="Prediction", color="black")
                ax.plot(self.t, self.data["Vel_4_1_2D"], label="Solution", color="gray")
                # ax.plot(
                #     self.data["t"],
                #     residual[:, 1] * 1e-6,
                #     label=r"Residual $\times 10^{-6}$",
                #     color="purple",
                #     linestyle="--",
                # )
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

    def get_variables(self, t, u):
        y, y_t, y_tt = [torch.zeros(4, t.shape[0])] * 3
        u_t, u_tt = self.get_u_derivatives(t, u)

        u3y, u4y, u3y_t, u4y_t, u3y_tt, u4y_tt = self.known_u_derivs(t)

        # Displacement
        y[0, :] = u[:, 0]
        y[1, :] = u3y
        y[2, :] = u[:, 1]
        y[1, :] = u4y

        # Velocity
        y_t[0, :] = u_t[:, 0]
        y_t[1, :] = u3y_t
        y_t[2, :] = u_t[:, 1]
        y_t[1, :] = u4y_t

        # Acceleration
        y_tt[0, :] = u_tt[:, 0]
        y_tt[1, :] = u3y_tt
        y_tt[2, :] = u_tt[:, 1]
        y_tt[1, :] = u4y_tt
        return y, y_t, y_tt

    def get_u_derivatives(
        self, t: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        u_t, u_tt = [u * 0] * 2
        for dim in range(int(u.shape[1])):
            u_t[:, dim] = dde.grad.jacobian(u, t, i=dim).squeeze()
            u_tt[:, dim] = dde.grad.hessian(u, t, component=dim).squeeze()
        return u_t, u_tt

    def load(self, t: torch.Tensor):
        x = t
        xp = self.data["t"]
        fp = self.data["load"]
        if isinstance(t, torch.Tensor):
            x = x.detach().cpu().numpy().squeeze()
            f = np.interp(x, xp, fp)
            f = torch.Tensor(f)
            ret = torch.zeros(t.shape[0], 4)
            ret[:, 3] = f
            ret = ret.permute((1, 0))
        else:
            f = np.interp(x, xp, fp).squeeze()
            ret = np.zeros((t.shape[0], 4))
            ret[:, 3] = f
            ret = ret.T
        return ret * 1e3  # convert kN -> N

    def known_u_derivs(self, t: torch.Tensor):
        x = t
        xp = self.data["t"]
        fp1 = self.data["Disp_3_2D"]
        fp2 = self.data["Disp_4_2D"]
        fp3 = self.data["Vel_3_2D"]
        fp4 = self.data["Vel_4_2D"]
        fp5 = self.data["Acc_3_2D"]
        fp6 = self.data["Acc_4_2D"]
        if isinstance(t, torch.Tensor):
            x = x.detach().cpu().numpy().squeeze()
            f1 = torch.Tensor(np.interp(x, xp, fp1))
            f2 = torch.Tensor(np.interp(x, xp, fp2))
            f3 = torch.Tensor(np.interp(x, xp, fp3))
            f4 = torch.Tensor(np.interp(x, xp, fp4))
            f5 = torch.Tensor(np.interp(x, xp, fp5))
            f6 = torch.Tensor(np.interp(x, xp, fp6))
        else:
            f1 = np.interp(x, xp, fp1).squeeze()
            f2 = np.interp(x, xp, fp2).squeeze()
            f3 = np.interp(x, xp, fp3).squeeze()
            f4 = np.interp(x, xp, fp4).squeeze()
            f5 = np.interp(x, xp, fp5).squeeze()
            f6 = np.interp(x, xp, fp6).squeeze()
        return f1, f2, f3, f4, f5, f6

    def get_residual(self, t, u):
        M = torch.Tensor(self.data["M"]).to("cuda")
        Kb = torch.Tensor(self.data["k_basis"]).to("cuda")
        a0, a1 = self.data["Damp_param"]
        F = self.load(t)
        K = Kb * self.E
        C = a0 * M + a1 * K

        y, y_t, y_tt = self.get_variables(t, u)

        # Whatever E is learned to be, it is actually 1e8 times that value
        residual = (
            self.U_MAX / self.T_MAX**2 * M @ y_tt
            - self.U_MAX / self.T_MAX * C @ y_t
            - F
        ) / 1e8 - self.U_MAX * K @ y
        residual = residual.permute((1, 0))
        return residual

    def force_magnitude(self, t):
        return np.interp(t, self.data["t"], self.data["load"]) * 1e3

import matplotlib.pyplot as plt
import deepxde as dde
import numpy as np
import torch

import data


def derivatives(x, y):
    y_t, y_tt = [y * 0] * 2
    for dim in range(y.shape[1]):
        y_t[:, dim] = dde.grad.jacobian(y, x, i=dim).squeeze()
        y_tt[:, dim] = dde.grad.hessian(y, x, component=dim).squeeze()
    return y_t, y_tt


def ode_system(x, y):
    """ODE system.
    M d2y/dx2 + C dy/dt + K y - F == 0
    """
    y_t, y_tt = derivatives(x, y)
    residual = (
        torch.mm(data.TORCH_M, y_tt.t())
        + torch.mm(data.TORCH_C, y_t.t())
        + torch.mm(data.TORCH_K, y.t())
        - data.F(x).t()
    )
    return residual.t()


def boundary(_, on_initial):
    return on_initial


def func(x):
    u = np.hstack(
        [
            np.interp(x.squeeze(), data.t, data.u[n, :]).reshape(-1, 1)
            for n in range(data.u.shape[0])
        ]
    )
    return u.reshape(-1, 4)


geom = dde.geometry.TimeDomain(data.t[0], data.t[-1])
ic = [dde.icbc.IC(geom, lambda x: 0, boundary, component=n) for n in range(4)]
pde = dde.data.PDE(geom, ode_system, ic, 35, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [4]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(pde, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

t = np.linspace(data.t[0], data.t[-1])

y_true = func(t.reshape(-1, 1))
y_pred = model.predict(t.reshape(-1, 1))

fig, ax = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
for dim in range(4):
    axes = ax[dim]

    axes.plot(t, y_true[:, dim], linestyle="--", color="gray", label="Solution")
    axes.plot(0, 0, linestyle="None", color="orange", label="Given Data")
    axes.plot(t, y_pred[:, dim], color="green")

    if dim < 2:
        axes.legend()
    axes.set_ylabel(r"$u_%d(t)$" % dim)
fig.suptitle("Model Prediction")
fig.supxlabel(r"Time, $t$")

plt.savefig("prediction.png", bbox_inches="tight")

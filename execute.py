# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN
import os
import argparse

torch.manual_seed(123)

plot_every = 100
parser = argparse.ArgumentParser()
parser.add_argument("--plot-every")
args = parser.parse_args()
plot_every = int(args.plot_every)

image_list = os.listdir("plots")
if image_list:
    [os.unlink(os.path.join("plots", image)) for image in image_list]


# %%
# Define callbacks
def epoch_logger(epoch, model, weighted_losses, **kw):
    if epoch % 1000 != 0:
        return
    print(
        "Epoch %d: Physics Loss %.4g, Node3YVel Loss %.4g, Node4YVel Loss %.4g, E = %.4g"
        % (epoch, weighted_losses[0], weighted_losses[1], weighted_losses[2], model.E())
    )


if not os.path.exists("plots"):
    os.mkdir("plots")


def plotter(epoch, model, data_t, u_pred_t, **kw):
    if epoch % plot_every != 0:
        return
    E = model.E()
    t = data_t.detach().cpu()
    v_pred = u_pred_t.detach().cpu()

    fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
    for dim in range(len(ax)):
        axes = ax[dim]

        axes.plot(t, v_pred[:, dim], label="Prediction")

        if dim == 1:
            axes.plot(
                t,
                model.node3_vel_y.detach().cpu(),
                marker=".",
                markersize=3,
                linestyle="None",
                label="Data",
            )
        elif dim == 3:
            axes.plot(
                t,
                model.node4_vel_y.detach().cpu(),
                marker=".",
                markersize=3,
                linestyle="None",
                label="Data",
            )
    fig.suptitle("Epoch = %d\nE = %.5g" % (epoch, E * 1e8))
    plt.savefig(os.path.join("plots", "%d.png" % epoch))
    plt.close()


# Define model
layers = [1] + 3 * [64] + [4]
sigmas = [1, 10, 50]
model = PINN(layers, sigmas)
model.load_ops_data()
model.compile(
    torch.optim.Adam(list(model.parameters()) + [model.a], lr=1e-4),
    callbacks=[epoch_logger, plotter],
    loss_weights=[1e-10, 1, 1],
)
model.train(iterations=int(2e6))

# %%

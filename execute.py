# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PINN
import utils
import os
import argparse

torch.manual_seed(123)

plot_every = 100
if not utils.is_notebook():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-every")
    args = parser.parse_args()
    plot_every = int(args.plot_every)

if not os.path.exists("media/plots"):
    os.makedirs("media/plots")

image_list = os.listdir("media/plots")
if image_list:
    [os.unlink(os.path.join("media/plots", image)) for image in image_list]


# %%
# Define callbacks
def epoch_logger(epoch, weighted_losses, **kw):
    if epoch % 1000 != 0:
        return
    print(
        "Epoch %d: Physics Loss %.4g, Node3YVel Loss %.4g, Node4YVel Loss %.4g"
        % (epoch, weighted_losses[0], weighted_losses[1], weighted_losses[2])
    )


def loss_logger(model, epoch, weighted_losses, **kw):
    if epoch == 0:
        model.loss_history = {
            "epochs": [],
            "physics_loss": [],
            "node3_yvel_loss": [],
            "node4_yvel_loss": [],
        }
    if epoch % 500 != 0:
        return

    model.loss_history["epochs"].append(epoch)
    model.loss_history["physics_loss"].append(weighted_losses[0].detach().cpu())
    model.loss_history["node3_yvel_loss"].append(weighted_losses[1].detach().cpu())
    model.loss_history["node4_yvel_loss"].append(weighted_losses[2].detach().cpu())
    utils.make_loss_plot(model.loss_history)


def plotter(model, epoch, data_t, u_pred_t, **kw):
    if epoch % plot_every != 0:
        return
    # E = model.E()
    t = data_t.detach().cpu()
    v_pred = u_pred_t.detach().cpu()

    fig, ax = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
    for dim in range(len(ax)):
        axes = ax[dim]

        axes.plot(t, v_pred[:, dim], label="Prediction")

        if dim == 0:
            axes.plot(t, model.node3_vel_x, linestyle="--", color="gray")
        elif dim == 1:
            axes.plot(
                t,
                model.node3_vel_y.detach().cpu(),
                marker=".",
                markersize=3,
                linestyle="None",
                label="Data",
            )
        elif dim == 2:
            axes.plot(t, model.node4_vel_x, linestyle="--", color="gray")
        elif dim == 3:
            axes.plot(
                t,
                model.node4_vel_y.detach().cpu(),
                marker=".",
                markersize=3,
                linestyle="None",
                label="Data",
            )
    fig.suptitle("Epoch = %d" % (epoch))
    plt.savefig(os.path.join("media/plots", "%d.png" % epoch))
    plt.close()


# Define model
layers = [1] + 5 * [64] + [4]
sigmas = [1, 10, 50]
model = PINN(layers, sigmas)
model.load_ops_data()
optimizer = torch.optim.Adam(list(model.parameters()), lr=1)
model.compile(
    optimizer,
    callbacks=[epoch_logger, loss_logger, plotter],
    loss_weights=[1e-11, 1e-2, 1e-3],
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=0.1, patience=10000, min_lr=1e-6
    ),
)
model.train(iterations=int(1e6))

utils.make_training_plot()
utils.make_loss_plot(model.loss_history)

# %%

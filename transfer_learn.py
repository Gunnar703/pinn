## Import Libraries

print("Importing libraries...")
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import deepxde as dde
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from generate_data import get_data
import argparse

from plotter_callback import PlotterCallback

print("Done.")

# Create the argument parser
parser = argparse.ArgumentParser()

# Add the command line argument
parser.add_argument(
    "--checkpoint-interval", type=int, help="Interval for saving plots/checkpoints."
)

# Parse the arguments
args = parser.parse_args()

# Access the value of the command line argument
checkpoint_interval = args.checkpoint_interval

# Print the checkpoint interval
print(f"Checkpoint interval: {checkpoint_interval}")

## Set hyperparameters
np.random.seed(123)
N_DEGREES_OF_FREEDOM = 4
device = "cuda"


## Ensure necessary files exist
def create_folder(fname):
    print(f"Necessary folder {dir[0]} not found. Creating...")
    os.mkdir(fname)
    print("Done.")


necessary_directories = [["model_files", "checkpoints"], ["plots", "training"]]
folders_created = []
for dir in necessary_directories:
    if not os.path.isdir(dir[0]):
        create_folder(dir[0])
        folders_created.append(dir[0])
    if not os.path.isdir(f"{dir[0]}/{dir[1]}"):
        create_folder(f"{dir[0]}/{dir[1]}")
        folders_created.append(f"{dir[0]}/{dir[1]}")
print("Created folders:" + "\n> ".join(["", *folders_created]))

## Get training data

print("Getting training data...")
data_folder = "data"
required_files = [
    "C",
    "Damp_param",
    "K",
    "M",
    "Vel_3_2D",
    "Vel_4_2D",
    "t",
    "load",
    "k_basis",
    "Y",
]
if not all([os.path.isfile(f"{data_folder}/{fn}.txt") for fn in required_files]):
    get_data(data_folder=data_folder)

max_rows = 290
data = {
    name: np.loadtxt(f"{data_folder}/{name}.txt", max_rows=max_rows)
    for name in required_files
}
print("Done.")


def force_magnitude(t):
    return np.interp(t, data["t"], data["load"]) * 1e3


force_idx = 3


def ode(t, u):
    y = u[:4].reshape(-1, 1)
    dy_dt = u[4:].reshape(-1, 1)

    force = force_magnitude(t)
    force_vec = np.zeros(4).reshape(-1, 1)
    force_vec[force_idx, 0] = -force

    # d2y_dt2 = np.linalg.inv(data["M"]) @ (force_vec - data["K"] @ y - data["C"] @ dy_dt)

    # Using k_basis
    d2y_dt2 = np.linalg.inv(data["M"]) @ (
        force_vec
        - (data["k_basis"] * data["Y"]) @ y
        - (data["Damp_param"][0] * data["M"] + data["Damp_param"][1] * data["K"])
        @ dy_dt
    )
    return np.hstack((dy_dt.squeeze(), d2y_dt2.squeeze()))


u0 = np.zeros(4 * 2)
tspan = (0, data["t"][-1])
sol = solve_ivp(ode, tspan, u0, max_step=1e-2)
tsol = sol.t
usol = sol.y[:4]
usol_derivative = sol.y[4:]

## Set up DeepXDE model
print("Setting up DeepXDE model...")


geometry = dde.geometry.TimeDomain(0, data["t"][-1])

# Define parameters
E_learned = dde.Variable(1.0)
# alpha_pi = dde.Variable(1.0)

# Define other tensors
M = torch.Tensor(data["M"])
K_basis = torch.Tensor(data["k_basis"])


# Define the ODE residual
def system(t, u):
    y = u
    y_t = torch.zeros_like(y).to(device)
    y_tt = torch.zeros_like(y).to(device)

    for dim in range(N_DEGREES_OF_FREEDOM):
        y_t[:, dim] = dde.grad.jacobian(u, t, i=dim, j=0).squeeze()
        y_tt[:, dim] = dde.grad.hessian(u, t, component=dim).squeeze()

    E = torch.abs(E_learned) * 1e7
    K = K_basis * E
    C = data["Damp_param"][0] * M + data["Damp_param"][1] * K

    F = np.zeros((t.shape[0], u.shape[1]))
    f_quasiscalar = force_magnitude(t.detach().cpu()).squeeze()
    F[:, force_idx] = -f_quasiscalar
    F = torch.Tensor(F)

    residual = (
        torch.mm(M, y_tt.permute((1, 0)))
        + torch.mm(torch.abs(C), y_t.permute((1, 0)))
        + torch.mm(torch.abs(K), y.permute((1, 0)))
        - F.permute((1, 0))
    ).permute((1, 0))

    return residual


def differentiate_u(t, u, component):
    return dde.grad.jacobian(u, t, i=component, j=0).reshape(-1, 1)


# B.C.'s on the velocity
bcs = [
    # Enforce y-velocity of node 3
    dde.icbc.boundary_conditions.PointSetOperatorBC(
        data["t"].reshape(-1, 1),
        data["Vel_3_2D"].reshape(-1, 1),
        (lambda t, u, X: differentiate_u(t, u, 1)),
    ),
    # Enforce y-velocity of node 4
    dde.icbc.boundary_conditions.PointSetOperatorBC(
        data["t"].reshape(-1, 1),
        data["Vel_4_2D"].reshape(-1, 1),
        (lambda t, u, X: differentiate_u(t, u, 3)),
    ),
    # Set initial x-velocity of node 3 to 0
    dde.icbc.boundary_conditions.PointSetOperatorBC(
        np.array([[0]]), np.array([[0]]), (lambda t, u, X: differentiate_u(t, u, 0))
    ),
    # Set initial x-velocity of node 4 to 0
    dde.icbc.boundary_conditions.PointSetOperatorBC(
        np.array([[0]]), np.array([[0]]), (lambda t, u, X: differentiate_u(t, u, 2))
    ),
]

pde_data = dde.data.PDE(
    geometry=geometry,
    pde=system,
    bcs=bcs,
    num_domain=500,
    num_boundary=2,
    num_test=10,
)

net = dde.nn.FNN(
    layer_sizes=[1] + 5 * [50] + [N_DEGREES_OF_FREEDOM],
    activation="tanh",
    kernel_initializer="Glorot uniform",
)
net.apply_output_transform(lambda x, y: y * (x))  # enforce starting at 0 as a hard b.c.

model = dde.Model(pde_data, net)
model.compile(
    "adam",
    lr=5e-5,
    external_trainable_variables=[E_learned],
    loss_weights=[1, 1, 1, 1, 1],
)

## Restore model
model.restore("model_files/data_fit.pt")


def differentiate_model_output(x, y):
    ret = torch.zeros_like(y)
    for i in range(4):
        ret[:, i] = dde.grad.jacobian(y, x, i=i, j=0).squeeze()
    return ret


## Verify model was loaded correctly
fig, axes = plt.subplots(4, 1, figsize=(8, 6))
v_pred = model.predict(data["t"].reshape(-1, 1), operator=differentiate_model_output)
for dim in range(4):
    ax = axes[dim]

    ax.plot(data["t"], v_pred[:, dim], label="Prediction", color="black")

    # Plot Solution Data
    ax.plot(
        tsol,
        usol_derivative[dim],
        label="Solution (RK-45)",
        color="gray",
    )

    # Plot given data
    if dim == 1:
        ax.plot(
            data["t"],
            data["Vel_3_2D"],
            label="Data (OPS)",
            marker="x",
            markersize=1,
            linestyle="None",
            color="orange",
        )
    elif dim == 3:
        ax.plot(
            data["t"],
            data["Vel_4_2D"],
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
fig.show()
plt.show()
plt.close()

plotter_callback = PlotterCallback(
    period=checkpoint_interval,
    filepath="plots/training",
    data=data,
    tsol=tsol,
    usol=usol_derivative,
)

checkpointer = dde.callbacks.ModelCheckpoint(
    "model_files/checkpoints/with_physics.pt", verbose=1, save_better_only=True
)
model.train(epochs=int(2e6), callbacks=[checkpointer, plotter_callback])
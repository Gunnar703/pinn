## Import Libraries

print("Importing libraries...")
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import deepxde as dde
from scipy import integrate
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from generate_data import get_data
import argparse

from plotter_callback import PlotterCallback

torch.backends.cuda.matmul.allow_tf32 = False

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

## Ensure output files (model_files/checkpoints, plots/training) are empty.
for path in ["/".join(entry) for entry in necessary_directories]:
    print(f"Checking {path}...")
    files = os.listdir(path)
    if not files:
        continue
    print(f"{path} not empty. Deleting contents...")
    for file in files:
        filepath = os.path.join(path, file)
        try:
            os.unlink(filepath)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (filepath, e))
    print("Done.")


## Get training data

print("Getting training data...")
data_folder = "data"
required_files = [
    "C",
    "Damp_param",
    "K",
    "M",
    "Vel_3_1_2D",
    "Vel_4_1_2D",
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
# Define domain
geometry = dde.geometry.TimeDomain(0, data["t"][-1])

# Define parameters
E_learned = dde.Variable(6.0)
# alpha_pi = dde.Variable(1.0)

# Define other tensors
M = torch.Tensor(data["M"])
K_basis = torch.Tensor(data["k_basis"])

## Numerically integrate/differentiate
du_t_1 = data["Vel_3_2D"]
du_t_3 = data["Vel_4_2D"]

u_1 = integrate.cumulative_trapezoid(du_t_1, data["t"], initial=0)
u_3 = integrate.cumulative_trapezoid(du_t_3, data["t"], initial=0)

du_tt_1, du_tt_3 = [np.zeros_like(du_t_1)] * 2
du_tt_1[1:] = (du_t_1[1:] - du_t_1[:-1]) / (data["t"][1:] - data["t"][:-1])
du_tt_3[1:] = (du_t_3[1:] - du_t_3[:-1]) / (data["t"][1:] - data["t"][:-1])

du_t_1 = torch.Tensor(du_t_1).to(device)
du_t_3 = torch.Tensor(du_t_3).to(device)
u_1 = torch.Tensor(u_1).to(device)
u_3 = torch.Tensor(u_3).to(device)
du_tt_1 = torch.Tensor(du_tt_1).to(device)
du_tt_3 = torch.Tensor(du_tt_3).to(device)

t_tens = torch.Tensor(data["t"])


def interp1d(lookup_x, lookup_y, lookup_val):
    np_x = lookup_x.detach().cpu().numpy()
    np_y = lookup_y.detach().cpu().numpy()
    np_val = lookup_val.detach().cpu().numpy()

    interp_val = np.interp(np_val, np_x, np_y)
    return torch.Tensor(interp_val)


# Define the ODE residual
def system(t, u):
    y = u
    y_t = torch.zeros_like(y).to(device)
    y_tt = torch.zeros_like(y).to(device)

    for dim in (0, 1):
        y_t[:, dim] = dde.grad.jacobian(u, t, i=dim, j=0).squeeze()
        y_tt[:, dim] = dde.grad.hessian(u, t, component=dim).squeeze()

    dy_t_1 = interp1d(t_tens, du_t_1, t.squeeze()).to(device)
    dy_t_3 = interp1d(t_tens, du_t_3, t.squeeze()).to(device)
    y_1 = interp1d(t_tens, u_1, t.squeeze()).to(device)
    y_3 = interp1d(t_tens, u_3, t.squeeze()).to(device)
    dy_tt_1 = interp1d(t_tens, du_tt_1, t.squeeze()).to(device)
    dy_tt_3 = interp1d(t_tens, du_tt_3, t.squeeze()).to(device)

    y = torch.concatenate(
        (
            y[:, 0].reshape(-1, 1),
            y_1.reshape(-1, 1),
            y[:, 1].reshape(-1, 1),
            y_3.reshape(-1, 1),
        ),
        1,
    )

    y_t = torch.concatenate(
        (
            y_t[:, 0].reshape(-1, 1),
            dy_t_1.reshape(-1, 1),
            y_t[:, 1].reshape(-1, 1),
            dy_t_3.reshape(-1, 1),
        ),
        1,
    )

    y_tt = torch.concatenate(
        (
            y_tt[:, 0].reshape(-1, 1),
            dy_tt_1.reshape(-1, 1),
            y_tt[:, 1].reshape(-1, 1),
            dy_tt_3.reshape(-1, 1),
        ),
        1,
    )

    E = torch.abs(E_learned) * 1e7
    K = K_basis * E
    C = data["Damp_param"][0] * M + data["Damp_param"][1] * K

    F = np.zeros((t.shape[0], 4))
    f_quasiscalar = force_magnitude(t.detach().cpu()).squeeze()
    F[:, force_idx] = -f_quasiscalar
    F = torch.Tensor(F)

    residual = (
        torch.mm(M, y_tt.permute((1, 0)))
        + torch.mm(torch.abs(C), y_t.permute((1, 0)))
        + torch.mm(torch.abs(K), y.permute((1, 0)))
        - F.permute((1, 0))
    ).permute((1, 0))

    return residual, torch.ones_like(residual) * torch.max(torch.abs(residual))


def differentiate_u(t, u, component):
    return dde.grad.jacobian(u, t, i=component, j=0).reshape(-1, 1)


# B.C.'s on the velocity
bcs = [
    # Enforce y-velocity of node 3
    dde.icbc.boundary_conditions.PointSetOperatorBC(
        np.array([[0]]),
        np.array([[0]]),
        (lambda t, u, X: differentiate_u(t, u, 0)),
    ),
    # Enforce y-velocity of node 4
    dde.icbc.boundary_conditions.PointSetOperatorBC(
        np.array([[0]]),
        np.array([[0]]),
        (lambda t, u, X: differentiate_u(t, u, 1)),
    ),
]

pde_data = dde.data.PDE(
    geometry=geometry,
    pde=system,
    bcs=bcs,
    num_domain=5000,
    num_boundary=2,
    num_test=10,
)

net = dde.nn.FNN(
    layer_sizes=[1] + 20 * [32] + [2],
    activation="tanh",
    kernel_initializer="Glorot uniform",
)
net.apply_output_transform(lambda x, y: y * (x))  # enforce starting at 0 as a hard b.c.

model = dde.Model(pde_data, net)

model.compile(
    "adam",
    lr=5e-4,
    external_trainable_variables=[E_learned],
)

if os.path.exists("model_files/train_further.pt"):
    model.restore("model_files/train_further.pt")

variable = dde.callbacks.VariableValue(
    [E_learned], period=checkpoint_interval, filename="variables.dat"
)

plotter_callback = PlotterCallback(
    period=checkpoint_interval,
    filepath="plots/training",
    data=data,
    tsol=tsol,
    usol=usol_derivative,
    E_learned=E_learned,
)

print("Done.")

losshistory, train_state = model.train(
    iterations=int(2e6), callbacks=[variable, plotter_callback]
)

print("Saving model...")
model.save("model_files/model")
dde.utils.saveplot(losshistory, train_state, issave=True, isplot=True)
print("Done.")

#### Print final E vector #####
print("Final learned E vector\n", "----------")
print("E = ", E_learned.detach())

print("True E vector\n", "----------")
print("E = ", data["Y"])

# ## Train with L-BFGS
# print("Training with L-BFGS")
# model.compile(
#     optimizer="L-BFGS",
#     external_trainable_variables=[E_learned],
#     loss_weights=[1e-10, 1e-10, 1, 1],
# )
# losshistory, train_state = model.train(callbacks=[variable, plotter_callback])
# model.save("model_files/model")
# dde.utils.saveplot(losshistory, train_state, issave=True, isplot=True)
# print("Done.")

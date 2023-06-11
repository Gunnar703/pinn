## Import Libraries

print("Importing libraries...")
import os
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
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

## Ensure output files (model_files/checkpoints, plots/training) are empty.
for path in ["/".join(entry) for entry in necessary_directories]:
    print(f"Checking {path}...")
    files = os.listdir(path)
    if not files:
        continue
    print(f"{path} not empty. Deleting contents...", end=" ")
    for file in files:
        filepath = os.path.join(path, file)
        try:
            os.unlink(filepath)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (filepath, e))
    print("Done.")


## Get training data

print("Getting training data...", end=" ")
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
    "Disp_3_2D",
    "Disp_4_2D",
]
if not all([os.path.isfile(f"{data_folder}/{fn}.txt") for fn in required_files]):
    get_data(data_folder=data_folder)

max_rows = 290
data = {
    name: np.loadtxt(f"{data_folder}/{name}.txt", max_rows=max_rows)
    for name in required_files
}
print("Done.")


## Non-dimensionalize data
t_char = max(data["t"])  # characteristic time measurement
u_char = max(
    np.max(data["Disp_3_2D"]), np.max(data["Disp_4_2D"])
)  # characteristic displacement measurement - chosen so that displacement is in [-1, 1]

data["t"] /= t_char
data["Disp_3_2D"] /= u_char
data["Disp_4_2D"] /= u_char


def non_dimensional_force(t_nd):
    """
    Interpolates non-dimensional time into a non-dimensionalized time
    vector and returns the corresponding force magnitude in Newtons.

    Args:
        t_nd (float): non-dimensional time at which to evaluate the force

    Returns:
        float: magnitude of force at specified non-dimensional time in Newtons
    """
    load = np.interp(t_nd, data["t"], data["load"]) * 1e3
    return load


data["force_idx"] = 3


## Get analytical solution
print("Obtaining Reference Solution...", end=" ")


def nd_ode(t, u):
    displacement_nd = u[:4].reshape(-1, 1)
    velocity_nd = u[4:].reshape(-1, 1)

    load = non_dimensional_force(t)
    F = np.zeros((4, 1))
    F[data["force_idx"]] = -load

    M = data["M"]
    K = data["Y"] * data["k_basis"]
    C = M * data["Damp_param"][0] + K * data["Damp_param"][1]

    acceleration_nd = (
        t_char**2
        / u_char
        * np.linalg.inv(M)
        @ (F - u_char * K @ displacement_nd - u_char / t_char * C @ velocity_nd)
    )
    return np.hstack((velocity_nd.squeeze(), acceleration_nd.squeeze()))


u0 = np.zeros(4 * 2)
tspan = (0, data["t"][-1])
sol = solve_ivp(nd_ode, tspan, u0, max_step=1e-3)
tsol = sol.t
usol = sol.y[:4]
usol_derivative = sol.y[4:]

## Plot analytical solution & training data
fig, axes = plt.subplots(4, 1, figsize=(8, 6))
fig.suptitle(r"Training Data ($E = %.5g$)" % data["Y"])
axes[-1].set_xlabel(r"Time ($t\ /\ t_c$)")
fig.text(
    0.04,
    0.5,
    r"Displacement ($\dot{u}\ t_c\ /\ u_c$)",
    va="center",
    rotation="vertical",
)

for dim in range(4):
    ax = axes[dim]
    ax.plot(tsol, usol[dim], label="RK45 (Solution)")

    if dim == 1:
        ax.plot(
            data["t"],
            data["Disp_3_2D"],
            label="OpenSees (Data)",
            linestyle="None",
            marker="x",
            markersize=2,
        )
    elif dim == 3:
        ax.plot(
            data["t"],
            data["Disp_4_2D"],
            label="OpenSees (Data)",
            linestyle="None",
            marker="x",
            markersize=2,
        )
    if dim == 1:
        ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 2.5))
plt.savefig("plots/training_data.png", bbox_inches="tight")
plt.close()
print("Done.")

print("Setting up DeepXDE Model...", end=" ")

## Define problem domain
geometry = dde.geometry.TimeDomain(0, data["t"][-1])

## Define ODE Parameters
E_learned = dde.Variable(6.0)  # set E pretty close to what it should be

## Convert other necessary elements to tensors
M = torch.Tensor(data["M"])
K_basis = torch.Tensor(data["k_basis"])


## Define the ODE Residual
def system(t_nd, u_nd):
    du_dt__nd = torch.zeros_like(u_nd).to(device)
    d2u_dt2__nd = torch.zeros_like(u_nd).to(device)

    E = torch.abs(E_learned) * 1e7  # Scale E up to the right magnitude
    K = K_basis * E
    C = data["Damp_param"][0] * M + data["Damp_param"][1] * K

    F = np.zeros((t_nd.shape[0], u_nd.shape[1]))
    load = non_dimensional_force(t_nd.detach().cpu()).squeeze()
    F[:, data["force_idx"]] = -load
    F = torch.Tensor(F).to(device)

    residual = (
        u_char / t_char**2 * torch.mm(M, d2u_dt2__nd.permute((1, 0)))
        + u_char / t_char * torch.mm(C, du_dt__nd.permute((1, 0)))
        + u_char * torch.mm(K, u_nd.permute((1, 0)))
        - F.permute((1, 0))
    ).permute((1, 0))
    return residual


def differentiate_u(t, u, component):
    """
    Differentiates u (non-dimensional) w.r.t. t (non-dimensional)

    Args:
        t (torch.FloatTensor): non-dimensional time
        u (torch.FloatTensor): non-dimensional NN prediction of displacement
        component (int): component of the output to differentiate

    Returns:
        torch.FloatTensor: derivative of the specified component of u w.r.t. t
    """
    return dde.grad.jacobian(u, t, i=component)


idx = np.unique(
    np.floor(np.sin(np.linspace(0, np.pi / 2, len(data["t"]))) * len(data["t"])) - 1
)
idx = [int(item) for item in idx]
vel_train_data = {
    "t": data["t"][idx],
    "Disp_3_2D": data["Disp_3_2D"][idx],
    "Disp_4_2D": data["Disp_4_2D"][idx],
}

new_idx = vel_train_data["t"].argsort()
vel_train_data = {
    "t": data["t"][new_idx],
    "Disp_3_2D": data["Disp_3_2D"][new_idx],
    "Disp_4_2D": data["Disp_4_2D"][new_idx],
}

# Position boundary conditions. Start at (0, 0) always.
bcs = [
    dde.icbc.boundary_conditions.PointSetBC(
        np.array([[0]]), np.array([[0]]), component=dim
    )
    for dim in (0, 2)
] + [
    dde.icbc.boundary_conditions.PointSetBC(
        data["t"].reshape(-1, 1), vel_train_data["Disp_3_2D"], component=1
    ),
    dde.icbc.boundary_conditions.PointSetBC(
        data["t"].reshape(-1, 1), vel_train_data["Disp_4_2D"], component=3
    ),
]

pde_data = dde.data.PDE(
    geometry=geometry,
    pde=system,
    bcs=bcs,
    num_domain=1000,
    num_boundary=2,
    num_test=10,
)

net = dde.nn.FNN(
    layer_sizes=[1] + 50 * [32] + [4],
    activation="tanh",
    kernel_initializer="Glorot uniform",
)

model = dde.Model(pde_data, net)
model.compile(
    "adam",
    lr=1e-4,
    external_trainable_variables=E_learned,
    loss_weights=[
        1e-4,  # residual/pde loss
        1e9,  # x-displacement of node 3 I.C.
        1e9,  # x-displacement of node 4 I.C.
        1e7,  # y-displacement of node 3
        1e7,  # y-displacement of node 4
    ],
)

variable_callback = dde.callbacks.VariableValue(
    E_learned, period=checkpoint_interval, filename="variables.dat"
)

plotter_callback = PlotterCallback(
    period=checkpoint_interval,
    filepath=f"plots/training",
    data=vel_train_data,
    tsol=tsol,
    usol=usol,
)

print("Done.")

losshistory, train_state = model.train(
    iterations=int(2e6), callbacks=[variable_callback, plotter_callback]
)

print("Saving model...")
model.save("model_files/model")
dde.utils.saveplot(losshistory, train_state, issave=True, isplot=True)
print("Done.")

#### Print final E vector #####
print("Final learned E vector\n", "----------")
print("E = \n", E_learned.detach())

print("True E vector\n", "----------")
print("EK = \n", data["Y"])

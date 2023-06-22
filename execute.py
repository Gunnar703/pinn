# %% [markdown]
# ## Import Libraries

# %%
import argparse
import matplotlib.pyplot as plt
from plotter_callback import PlotterCallback
import deepxde as dde
import numpy as np
import os
import torch
from scipy.integrate import solve_ivp
from generate_data import get_data

torch.backends.cuda.matmul.allow_tf32 = False
checkpoint_interval = 10_000

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

# %% [markdown]
# ## Import Data

# %%
fn = "data"
if not os.path.exists(fn):
    get_data()
dirs = os.listdir(fn)
data = {
    name.split(".")[0]: np.loadtxt(os.path.join(fn, name), max_rows=290)
    for name in dirs
}

# Use FDM to get accelerations
acc_3 = (data["Vel_3_2D"][1:] - data["Vel_3_2D"][:-1]) / (
    data["t"][1:] - data["t"][:-1]
)
acc_4 = (data["Vel_4_2D"][1:] - data["Vel_4_2D"][:-1]) / (
    data["t"][1:] - data["t"][:-1]
)

data["Acc_3_2D"] = np.zeros_like(data["t"])
data["Acc_4_2D"] = np.zeros_like(data["t"])

data["Acc_3_2D"][1:] = acc_3
data["Acc_4_2D"][1:] = acc_4

# Normalize Data
T_MAX = data["t"].max()
U_MAX = max(data["Disp_3_2D"].max(), data["Disp_4_2D"].max())

data["t"] /= T_MAX
for name in data:
    if "Vel" in name or "Disp" in name:
        data[name] /= U_MAX

# For convenience
a0, a1 = data["Damp_param"]

# Get Constant Tensors ready
M = torch.Tensor(data["M"]).to("cuda")
Kb = torch.Tensor(data["k_basis"]).to("cuda")


# Define interpolation of F. Returns M x N_DIM tensor.
def load(t: torch.Tensor | float):
    x = t
    xp = data["t"]
    fp = data["load"]
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


# Define interpolation of u, u_t, u_tt
def known_u_derivs(t: torch.Tensor | float):
    x = t
    xp = data["t"]
    fp1 = data["Disp_3_2D"]
    fp2 = data["Disp_4_2D"]
    fp3 = data["Vel_3_2D"]
    fp4 = data["Vel_4_2D"]
    fp5 = data["Acc_3_2D"]
    fp6 = data["Acc_4_2D"]
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


# %% [markdown]
# ## Define the PDE

# %%
# Geometry - just an interval
geom = dde.geometry.TimeDomain(data["t"].min(), data["t"].max())


# Helper function
def get_u_derivatives(t: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    u_t, u_tt = [u * 0] * 2
    for dim in range(int(u.shape[1])):
        u_t[:, dim] = dde.grad.jacobian(u, t, i=dim).squeeze()
        u_tt[:, dim] = dde.grad.hessian(u, t, component=dim).squeeze()
    return u_t, u_tt


# Learnable parameter/s
E = dde.Variable(0.6)


# Helper function
def get_variables(t, u):
    y, y_t, y_tt = [torch.zeros(4, t.shape[0])] * 3
    u_t, u_tt = get_u_derivatives(t, u)

    u3y, u4y, u3y_t, u4y_t, u3y_tt, u4y_tt = known_u_derivs(t)

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


# ODE definition
def ode_sys(t, u):
    F = load(t)
    K = Kb * E
    C = a0 * M + a1 * K

    y, y_t, y_tt = get_variables(t, u)

    # Whatever E is learned to be, it is actually 1e8 times that value
    residual = (
        U_MAX / T_MAX**2 * M @ y_tt - U_MAX / T_MAX * C @ y_t - F
    ) / 1e8 - U_MAX * K @ y
    residual = residual.permute((1, 0))
    return residual


# Boundary conditions definition
def differentiate_output(t, u, component, order):
    if order == 1:
        return dde.grad.jacobian(u, t, i=component)
    return dde.grad.hessian(u, t, component=component)


t_data = data["t"].reshape(-1, 1)
zero_vector = np.array([[0]])

# IC for unknown dimensions
v0 = [
    dde.icbc.PointSetOperatorBC(
        zero_vector, zero_vector, lambda t, u, X: differentiate_output(t, u, i, 1)
    )
    for i in (0, 1)
]

pde = dde.data.PDE(
    geom,
    ode_sys,
    v0,
    num_domain=400,
    num_boundary=2,
)

# %% [markdown]
# ## Define the Network

# %%
# Callbacks
variable = dde.callbacks.VariableValue(
    var_list=[E], period=checkpoint_interval, filename="variables.dat"
)

plotter_callback = PlotterCallback(
    period=checkpoint_interval,
    filepath="plots/training",
    data=data,
    E=E,
    t_max=T_MAX,
    u_max=U_MAX,
    plot_residual=False,
)

net = dde.nn.FNN(
    layer_sizes=[1] + 8 * [100] + [2],
    activation="tanh",
    kernel_initializer="Glorot uniform",
)
model = dde.Model(pde, net)

model.compile(optimizer="adam", lr=1e-5, external_trainable_variables=E)


# %% [markdown]
# ## Make sure directories exist and are empty

# %%
necessary_directories = [["model_files", "checkpoints"], ["plots", "training"]]
folders_created = []
for dir in necessary_directories:
    if not os.path.isdir(dir[0]):
        os.mkdir(dir[0])
        folders_created.append(dir[0])
    if not os.path.isdir(f"{dir[0]}/{dir[1]}"):
        os.mkdir(f"{dir[0]}/{dir[1]}")
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

# %%
losshistory, train_state = model.train(
    iterations=1_000_000, callbacks=[variable, plotter_callback]
)

dde.saveplot(losshistory, train_state)
model.save("model_files/model")

# %%
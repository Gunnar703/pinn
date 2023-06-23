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
def load(t: torch.Tensor):
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


# ODE definition
def ode_sys(t, u):
    F = load(t)
    K = Kb * torch.abs(E)
    C = a0 * M + a1 * K

    y_t, y_tt = get_u_derivatives(t, u)
    y = u.permute((1, 0))
    y_t = y_t.permute((1, 0))
    y_tt = y_tt.permute((1, 0))

    # Whatever E is learned to be, it is actually 1e8 times that value
    U = y * U_MAX
    DU_DT = y_t * U_MAX / T_MAX
    D2U_DT2 = y_tt * U_MAX / T_MAX**2

    mass_term = M @ D2U_DT2 / 1e8
    damp_term = (a1 * Kb * E) @ DU_DT + (a0 * M) / 1e8 @ DU_DT
    stiff_term = (Kb * E) @ U
    force_term = F / 1e8
    residual = mass_term + damp_term + stiff_term - force_term
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

# Known dimensions
vi = [
    dde.icbc.PointSetOperatorBC(
        t_data,
        data["Vel_3_2D"].reshape(-1, 1),
        lambda t, u, X: differentiate_output(t, u, 1, 1),
    ),
    dde.icbc.PointSetOperatorBC(
        t_data,
        data["Vel_4_2D"].reshape(-1, 1),
        lambda t, u, X: differentiate_output(t, u, 3, 1),
    ),
]

# Position BC
xi = [
    dde.icbc.PointSetBC(t_data, data["Disp_3_2D"].reshape(-1, 1), component=1),
    dde.icbc.PointSetBC(t_data, data["Disp_4_2D"].reshape(-1, 1), component=3),
]

# Acceleration BC
ai = [
    dde.icbc.PointSetOperatorBC(
        t_data,
        data["Acc_3_2D"].reshape(-1, 1),
        lambda t, u, X: differentiate_output(t, u, 1, 2),
    ),
    dde.icbc.PointSetOperatorBC(
        t_data,
        data["Acc_4_2D"].reshape(-1, 1),
        lambda t, u, X: differentiate_output(t, u, 3, 2),
    ),
]

pde = dde.data.PDE(
    geom,
    ode_sys,
    vi + xi,
    num_domain=1000,
    num_boundary=2,
)

# %% [markdown]
# ## Define the Network

# %%
# Callbacks
variable = dde.callbacks.VariableValue(
    var_list=[E], period=checkpoint_interval, filename="out_files/variables.dat"
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
    layer_sizes=[1] + 8 * [100] + [4],
    activation="tanh",
    kernel_initializer="Glorot uniform",
)
model = dde.Model(pde, net)

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

if not os.path.exists("out_files"):
    os.path.mkdir("out_files")

# %% [markdown]
# ## Train the Model

# %%
X = geom.random_points(1_000)
f = model.predict(X, operator=ode_sys)
print("X:", X.shape)
print("f mean:", f.mean(axis=1).shape)

model.compile(optimizer="adam", lr=1e-5, external_trainable_variables=E)
model.train(iterations=50_000, callbacks=[plotter_callback, variable])
model.compile("L-BFGS", external_trainable_variables=E)
model.train(callbacks=[plotter_callback, variable])

X = geom.random_points(1_000)
err = 1
while err > 0.005:
    f = model.predict(X, operator=ode_sys).mean(axis=1)
    err_eq = np.absolute(f)
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))
    print(f"E = {E}")

    x_id = np.argmax(err_eq)
    print("Adding new point:", X[x_id], "\n")
    pde.add_anchors(X[x_id])
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-6, patience=2000)
    model.compile("adam", lr=1e-5, external_trainable_variables=E)
    model.train(
        iterations=20_000,
        disregard_previous_best=True,
        callbacks=[early_stopping, plotter_callback, variable],
    )
    model.compile("L-BFGS", external_trainable_variables=E)
    losshistory, train_state = model.train(callbacks=[plotter_callback, variable])

# %% [markdown]
# ## Save the model and loss history

# %%
dde.saveplot(
    losshistory,
    train_state,
    loss_fname="out_files/loss.dat",
    train_fname="out_files/train.dat",
    test_fname="out_files/test.dat",
)
model.save("model_files/model")

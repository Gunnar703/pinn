# %% [markdown]
# ## Four Degree-of-Freedom System
# 
# ### Known
# - $ \left[ M \right] $ 
# - $ a_0,\ a_1 $
# - $ \left[ C \right] = a_0 \left[ M \right] + a_1 \left[ K \right] $
# - $ K_{ij} \geq 0\ \forall (i, j) \in \mathbb{N} \times \mathbb{N} $
# - $ \left[ K \right] $ is sparse
# - $ \left[ K \right] = \sum_{i=1}^4 \left[ \mathbb{K}_\text{basis} \right]_i \cdot E_i $
# 
# ### Unknown
# - $ \mathbb{E} = \bigcup_{i=1}^4 E_i $
# - $ \alpha_\pi $
# 
# ### Constraints
# - $ \mathcal{J}_\mathcal{D} = \frac{1}{2} \sum_{i=1,2} \left( \hat{u}_i - u_i \right)^2 $ (data loss)
# - $ E_{ij} \geq 0\ \forall (i, j) \in \mathbb{N} \times \mathbb{N} $ (hard constraint)
# - $ \mathcal{J}_\pi = \alpha_\pi \mathcal{L}_2\left( \left[ M \right]\left[ \ddot{u} \right] + \left[ C \right]\left[ \dot{u} \right] + \left[ K \right]\left[ u \right] - \left[ f(t) \right] \right) $
# - $ \mathcal{J}_\mathcal{S} = \mathcal{L}_1\left( \left[ K \right] \right) $ (sparsity enforcement, not used here because the 4DOF K-matrix is not actually sparse)
# 
# ### Definitions
# $\mathcal{L}_1$: Taxicab norm\
# $\mathcal{L}_2$: Euclidiean norm

# %%
print("Importing libraries...")
## Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import deepxde as dde
import torch
import torch.nn as nn
from scipy.integrate import odeint
print("Done.")

## Set hyperparameters
np.random.seed(123)
N_DEGREES_OF_FREEDOM = 4
N_COLLOC_POINTS = 50
device = 'cuda'

# %%
print("Defining training data...")
## Define Known Values
omega1 = 2 * np.pi * 1.5 #1.5 hz first mode
omega2 = 2 * np.pi * 14
damp1 = 0.01
damp2 = 0.02
a0 = ( 2 * damp1 * omega1 * (omega2**2) - 2 * damp2 * omega2 * (omega1**2) ) / ( (omega2**2) -(omega1**2) )
a1 = ( 2 * damp2 * omega2 - 2 * damp1 * omega1 ) / ( (omega2**2) -(omega1**2) )

m = np.diag( np.ones(N_DEGREES_OF_FREEDOM) )
e = np.random.rand(N_DEGREES_OF_FREEDOM)
print(f"True E: {e}")

first_diag = int( np.floor(N_DEGREES_OF_FREEDOM*3/4) )
k_basis = np.transpose(np.array([
    np.diag(np.random.rand(N_DEGREES_OF_FREEDOM))
    +
    np.diag(np.random.rand(first_diag), k=N_DEGREES_OF_FREEDOM-first_diag) * 0.6
    +
    np.diag(np.random.rand(first_diag), k=-(N_DEGREES_OF_FREEDOM-first_diag)) * 0.6
    for _ in range(N_DEGREES_OF_FREEDOM)
]), axes=(1, 0, 2))

k = np.dot(e, k_basis)
c = a0 * m + a1 * k

force_index = 1
def np_force(t):
    force_mask = np.zeros( N_DEGREES_OF_FREEDOM ).reshape(-1, 1)
    force_mask[force_index] = 1
    return np.exp(-(t-np.pi)**2) * np.sin(2*np.pi*t) * force_mask

# %%
## Solve ODE
def ode(u, t):
    y    = u[ 0 : N_DEGREES_OF_FREEDOM ].reshape(-1, 1)
    y_t  = u[ N_DEGREES_OF_FREEDOM : ].reshape(-1, 1)
            
    y_tt = np.linalg.inv(m) @ (
        np_force(t)
        -
        c @ y_t
        -
        k @ y
    )
    return np.array( list( y_t.squeeze() ) + list(y_tt.squeeze()) )

u0 = np.zeros( N_DEGREES_OF_FREEDOM * 2 )
t  = np.linspace( 0, 4 * np.pi, 150 )

sol = odeint(ode, u0, t)
u   = sol[:, :N_DEGREES_OF_FREEDOM]
u_t = sol[:, N_DEGREES_OF_FREEDOM:]

# %%
## Screen out training data
# time_indices   = np.arange( 0, len(t), len(t) // N_COLLOC_POINTS )
time_indices   = np.arange( 0, len(t), 1 )
sensor_indices = [1, 3]

tdata = t[time_indices]
udata = u[time_indices]  # this data includes all dimensions, not just the sensor dimensions
print("Done.")

print("Plotting...")
## Plot
fig, ax = plt.subplots(N_DEGREES_OF_FREEDOM, 1, sharex=True)
plt.suptitle("Solution and Data")
for dim in range(N_DEGREES_OF_FREEDOM):
    ax[dim].plot(t, u[:, dim], label="Solution")
    if dim in sensor_indices:
        ax[dim].plot(tdata, udata[:, dim], label="Data", linestyle="None", marker=".")
    else:
        ax[dim].plot(tdata[0], udata[0, dim], label="Data", linestyle="None", marker=".")
    ax[dim].set_xlabel(r"$t$")
    ax[dim].set_ylabel(r"$u_{}(t)$".format(dim + 1))
    if dim == 0:
        ax[dim].legend(loc="upper right", ncol=2)
plt.savefig("plots/training_data.png")
plt.close()
print("Done.")

# %%
## Set up DeepXDE model
print("Setting up DeepXDE model...")
# Define domain
geometry = dde.geometry.TimeDomain( t[0], t[-1] )

# Define forcing function
def pt_force(t):
    return torch.cat(
        [
            (torch.exp(-(t-np.pi)**2) * torch.sin(2*np.pi*t)).view(1, -1) if dim == force_index else (t * 0).reshape(1, -1)
            for dim in range(N_DEGREES_OF_FREEDOM)
        ],
        axis = 0
    )

# Define parameters
E        = dde.Variable( np.ones_like( e ), dtype=torch.float32 )
alpha_pi = dde.Variable(1.0)

# Define other tensors
M = torch.Tensor(m)
K_basis = torch.Tensor(k_basis)

# Define the ODE residual
def system (t, u):
    y    = u
    y_t  = torch.zeros_like( y ).to(device)
    y_tt = torch.zeros_like( y ).to(device)
    
    for dim in range( N_DEGREES_OF_FREEDOM ):
        y_t [:, dim] = dde.grad.jacobian( u, t, i=dim, j=0 ).squeeze()
        y_tt[:, dim] = dde.grad.hessian ( u, t, component=dim ).squeeze()
    
    K = torch.matmul( torch.abs(E), K_basis )
    C = a0 * M + a1 * K
            
    residual = (
        torch.mm( M, y_tt.permute((1, 0)) )
        +
        torch.mm( torch.abs(C), y_t.permute((1, 0)) )
        +
        torch.mm( torch.abs(K), y.permute((1, 0)) )
        -
        pt_force(t)
    ).permute((1, 0))
    return max(alpha_pi, 1.0) * residual

bcs = [
    ( 
        dde.icbc.boundary_conditions.PointSetBC( tdata.reshape(-1, 1), udata[:, dim].reshape(-1, 1), component=dim )
    ) if (dim in sensor_indices) else (
        dde.icbc.boundary_conditions.PointSetBC( tdata[0].reshape(-1, 1), udata[0, dim].reshape(-1, 1), component=dim )
    )
    for dim in range(N_DEGREES_OF_FREEDOM)
]

data = dde.data.PDE(
    geometry     = geometry,
    pde          = system,
    bcs          = bcs,
    num_domain   = 10000,
    num_boundary = 2,
    num_test     = 5
)

net = dde.nn.FNN(
    layer_sizes        = [1] + 20*[32] + [N_DEGREES_OF_FREEDOM],
    activation         = "tanh",
    kernel_initializer = "Glorot uniform"
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4, external_trainable_variables=[E, alpha_pi])

variable = dde.callbacks.VariableValue(
  list(torch.abs(E)) + [max(alpha_pi, 1.0)], period=1000, filename="variables.dat"
)

checkpoint = dde.callbacks.ModelCheckpoint("model_files/checkpoints/model", period=10_000)

epoch = 0
def plot():
    global epoch
    epoch += 1
    if checkpoint.epochs_since_last_save + 1 < checkpoint.period: return
    upred = model.predict(t.reshape(-1, 1))
    _, ax = plt.subplots(N_DEGREES_OF_FREEDOM, 1, sharex=True)
    plt.suptitle(f"Epoch {epoch}")
    for dim in range(N_DEGREES_OF_FREEDOM):
        ax[dim].plot(t, u[:, dim], label="Solution", color='blue')
        if dim in sensor_indices:
            ax[dim].plot(tdata, udata[:, dim], label="Data", linestyle="None", marker=".", color='orange')
        else:
            ax[dim].plot(tdata[0], udata[0, dim], label="Data", linestyle="None", marker=".", color='orange')
        ax[dim].plot(t, upred[:, dim], label="Prediction", color='green')
        ax[dim].set_xlabel(r"$t$")
        ax[dim].set_ylabel(r"$u_{}(t)$".format(dim + 1))
        if dim == 0:
            ax[dim].legend(loc="upper right", ncol=2)
    plt.savefig(f"plots/training/epoch_{epoch}_prediction.png")
    plt.close()

checkpoint.on_epoch_begin = plot
print("Done.")
losshistory, train_state = model.train(iterations=2_000_000, callbacks=[variable, checkpoint])

# %%
print("Saving model...")
model.save("model_files/model")
dde.utils.saveplot(losshistory, train_state, issave=True, isplot=True)
print("Done.")

#### Print final E vector #####
print("Final learned E vector\n", "----------")
print("E = \n", E.detach())

print("True E vector\n", "----------")
print("EK = \n", e)
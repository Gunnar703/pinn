##### Imports and global settings #####
import numpy as np
import deepxde as dde
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import torch

device = 'cuda'
np.random.seed(123)
RUN_NAME = '4_dof_ydata_only'
N_COLLOC_POINTS = 15
N_DIMS = 4

print("Necessary libraries imported.")

##### Solve ODE #####

print("Solving forward problem...")
# Define the problem
m = np.diag( np.ones( N_DIMS ) )
c = np.diag( np.ones( N_DIMS ) )
k = np.diag( np.ones( N_DIMS ) ) * 9 + np.ones( (N_DIMS, N_DIMS) )

u0 = np.zeros( N_DIMS * 2 )
t  = np.linspace( 0, 4 * np.pi, 100 )

force_index = 1
def np_force(t):
    force_mask  = np.zeros( N_DIMS ).reshape(-1, 1)
    force_mask[force_index] = 1
    return np.exp(-t) * force_mask

def pt_force(t):
    return torch.cat(
        [
            torch.exp(-t).view(1, -1) if dim == force_index else (t * 0).reshape(1, -1)
            for dim in range(N_DIMS)
        ],
        axis = 0
    )

def ode(u, t):
    y    = u[ 0 : N_DIMS ].reshape(-1, 1)
    y_t  = u[ N_DIMS : ].reshape(-1, 1)
            
    y_tt = np.linalg.inv(m) @ (
        np_force(t)
        -
        c @ y_t
        -
        k @ y
    )
    return np.array( list( y_t.squeeze() ) + list(y_tt.squeeze()) )

# Solve
sol = odeint(ode, u0, t)
u   = sol[:, :N_DIMS]
u_t = sol[:, N_DIMS:]

##### Screen out training data #####
time_indices   = np.arange( 0, len(t), len(t) // N_COLLOC_POINTS )
sensor_indices = [1, 3]

tdata = t[time_indices]
udata = u[time_indices]  # this data includes all dimensions, not just the sensor dimensions
print("Done.")

print("Plotting and saving training data...")
# Plot training data
fig, ax = plt.subplots(N_DIMS, 1, sharex=True)
plt.suptitle("Solution and Data")
for dim in range(N_DIMS):
    ax[dim].plot(t, u[:, dim], label="Solution")
    if dim in sensor_indices:
        ax[dim].plot(tdata, udata[:, dim], label="Data", linestyle="None", marker=".")
    ax[dim].set_xlabel(r"$t$")
    ax[dim].set_ylabel(r"$u_{}(t)$".format(dim + 1))
    if dim == 0:
        ax[dim].legend(loc="upper right", ncol=2)
plt.savefig('plots/training_data.png')  # Save to plots folder
plt.close()
print("Done.")

##### Define the problem #####
print("Defining inverse problem in DeepXDE...")
# Define domain
geometry = dde.geometry.TimeDomain( t[0], t[-1] )

# Define parameters
M = torch.Tensor( m ).to( device )
C = dde.Variable( np.ones_like( c ), dtype=torch.float32 )
K = dde.Variable( np.ones_like( k ), dtype=torch.float32 )

variable_list = []
for i in range(N_DIMS):
    for j in range(N_DIMS):
        variable_list.append( C[i, j] )
        variable_list.append( K[i, j] )

# Define the ode residual
def system (t, u):
    y    = u
    y_t  = torch.zeros_like( y ).to(device)
    y_tt = torch.zeros_like( y ).to(device)
    
    for dim in range( N_DIMS ):
        y_t [:, dim] = dde.grad.jacobian( u, t, i=dim, j=0 ).squeeze()
        y_tt[:, dim] = dde.grad.hessian ( u, t, component=dim ).squeeze()
            
    residual = (
        torch.mm( M, y_tt.permute((1, 0)) )
        +
        torch.mm( torch.abs(C), y_t.permute((1, 0)) )
        +
        torch.mm( torch.abs(K), y.permute((1, 0)) )
        -
        pt_force(t)
    ).permute((1, 0))
    return residual

bcs = [
    dde.icbc.boundary_conditions.PointSetBC( tdata.reshape(-1, 1), udata[:, dim].reshape(-1, 1), component=dim )
    for dim in sensor_indices
]

data = dde.data.PDE(
    geometry     = geometry,
    pde          = system,
    bcs          = bcs,
    num_domain   = 5000,
    num_boundary = 2,
    num_test     = 5
)

net = dde.nn.FNN(
    layer_sizes        = [1] + 20*[32] + [N_DIMS],
    activation         = "tanh",
    kernel_initializer = "Glorot uniform"
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-4, external_trainable_variables=[C, K])

variable = dde.callbacks.VariableValue(
  variable_list, period=1000, filename="variables.dat"
)

checkpoint = dde.callbacks.ModelCheckpoint("model_files/checkpoints/model", period=10_000)

epoch = -1
def plot():
    global epoch
    epoch += 1
    if checkpoint.epochs_since_last_save + 1 < checkpoint.period: return
    upred = model.predict(t.reshape(-1, 1))
    fig, ax = plt.subplots(N_DIMS, 1, sharex=True)
    plt.suptitle(f"Epoch {epoch}")
    for dim in range(N_DIMS):
        ax[dim].plot(t, u[:, dim], label="Solution", color='blue')
        if dim in sensor_indices:
            ax[dim].plot(tdata, udata[:, dim], label="Data", linestyle="None", marker=".", color='orange')
        ax[dim].plot(t, upred[:, dim], label="Prediction", color='green')
        ax[dim].set_xlabel(r"$t$")
        ax[dim].set_ylabel(r"$u_{}(t)$".format(dim + 1))
        if dim == 0:
            ax[dim].legend(loc="upper right", ncol=2)
    plt.savefig(f"plots/train_history/epoch_{epoch}_prediction.png")
    plt.close(plt.gcf())

checkpoint.on_epoch_begin = plot

losshistory, train_state = model.train(iterations=2_000_000, callbacks=[variable, checkpoint])

print("Plotting final model prediction and saving model...")
##### Plot Final Prediction #####
upred = model.predict(t.reshape(-1, 1))

fig, ax = plt.subplots(N_DIMS, 1, sharex=True)
plt.suptitle(f"{RUN_NAME}")
for dim in range(N_DIMS):
    ax[dim].plot(t, u[:, dim], label="Solution")
    if dim in sensor_indices:
        ax[dim].plot(tdata, udata[:, dim], label="Data", linestyle="None", marker=".")
    ax[dim].plot(t, upred[:, dim], label="Prediction")
    ax[dim].set_xlabel(r"$t$")
    ax[dim].set_ylabel(r"$u_{}(t)$".format(dim + 1))
    if dim == 0:
        ax[dim].legend(loc="upper right", ncol=2)

plt.savefig(f"plots/{RUN_NAME}.png")

##### Save final model ##### 
model.save(f"model_files/{RUN_NAME}")
print("Done.")

#### Print final C and K matrices #####
print("Final learned C and K matrices\n", "----------")
print("C = \n", C.detach())
print("K = \n", K.detach())

print("True C and K matrices\n", "----------")
print("C = \n", c)
print("K = \n", k)
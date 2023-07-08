from ops import get_data
import torch
import numpy as np
import os

device = "cuda"

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
    "Disp_3_1_2D",
    "Disp_4_1_2D",
    "Disp_3_2D",
    "Disp_4_2D",
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

## Convenience defs
M = data["M"]
C = data["C"]
K = data["K"]
u_t = np.array(
    [
        data["Vel_3_1_2D"],
        data["Vel_3_2D"],
        data["Vel_4_1_2D"],
        data["Vel_4_2D"],
    ]
)
u = np.array(
    [
        data["Disp_3_1_2D"],
        data["Disp_3_2D"],
        data["Disp_4_1_2D"],
        data["Disp_4_2D"],
    ]
)
t = data["t"]
load = data["load"]
k_basis = data["k_basis"]
Y = data["Y"]
a0, a1 = data["Damp_param"]

T_MAX = max(t)
U_MAX = max(u)

t_norm = t / T_MAX
u_norm = u / U_MAX
u_t_norm = u_t * T_MAX / U_MAX

# torch versions
TORCH_M = torch.Tensor(M).to(device)
TORCH_C = torch.Tensor(C).to(device)
TORCH_K = torch.Tensor(K).to(device)
TORCH_k_basis = torch.Tensor(k_basis).to(device)


def F(t):
    F_ = np.zeros((t.shape[0], 4))
    f_quasiscalar = force_magnitude(t.detach().cpu()).squeeze()
    F_[:, force_idx] = -f_quasiscalar
    F_ = torch.Tensor(F_)
    return F_.to(device)

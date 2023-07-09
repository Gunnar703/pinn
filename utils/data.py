import openseespy.opensees as ops
import os as os
import numpy as np
import os

import torch


def get_data(data_folder="data", nu=0.3, Vs=150):
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    ops.node(1, 0, 0)
    ops.node(2, 5, 0)
    ops.node(3, 5, 5)
    ops.node(4, 0, 5)

    Rho = 2 * 10**3
    # Vs = 150  # set as default arguments
    # nu = 0.3  # set as default arguments
    Vp = Vs * np.sqrt(2 * (1 - nu) / (1 - 2 * nu))
    G = Vs**2 * Rho
    Y = 2 * G / (1 + nu)  # only one element, so only one Y

    np.savetxt(f"{data_folder}/Y.txt", np.array([Y]))

    ops.nDMaterial("ElasticIsotropic", 100, Y, nu, Rho)
    ops.element("quad", 1000, 1, 2, 3, 4, 1.0, "PlaneStrain", 100)

    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)

    ## Velocity Recorders
    ops.recorder(
        "Node", "-file", f"{data_folder}/Vel_3_2D.txt", "-node", 3, "-dof", 2, "vel"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Vel_4_2D.txt", "-node", 4, "-dof", 2, "vel"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Vel_3_1_2D.txt", "-node", 3, "-dof", 1, "vel"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Vel_4_1_2D.txt", "-node", 4, "-dof", 1, "vel"
    )

    ## Displacement Recorders
    ops.recorder(
        "Node", "-file", f"{data_folder}/Disp_3_2D.txt", "-node", 3, "-dof", 2, "disp"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Disp_4_2D.txt", "-node", 4, "-dof", 2, "disp"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Disp_3_1_2D.txt", "-node", 3, "-dof", 1, "disp"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Disp_4_1_2D.txt", "-node", 4, "-dof", 1, "disp"
    )

    omega1 = 2 * np.pi * 1.5  # 1.5 hz first mode
    omega2 = 2 * np.pi * 14
    damp1 = 0.05
    damp2 = 0.1
    a0 = (2 * damp1 * omega1 * (omega2**2) - 2 * damp2 * omega2 * (omega1**2)) / (
        (omega2**2) - (omega1**2)
    )
    a1 = (2 * damp2 * omega2 - 2 * damp1 * omega1) / ((omega2**2) - (omega1**2))
    np.savetxt(f"{data_folder}/Damp_param.txt", np.array([a0, a1]))

    time_history = f"{data_folder}/t.txt"
    load_history = f"{data_folder}/load.txt"
    ops.timeSeries(
        "Path", 300, "-fileTime", time_history, "-filePath", load_history, "-factor", 1
    )
    ops.pattern("Plain", 400, 300)
    ops.load(4, 0, -1000)

    ops.constraints("Transformation")
    ops.rayleigh(a0, a1, 0.0, 0.0)
    ops.integrator("Newmark", 0.5, 0.25)
    ops.numberer("RCM")
    ops.system("SparseGeneral")
    # ops.constraints('Transformation')
    ops.test("NormDispIncr", 1e-12, 800, 1)
    ops.algorithm("ModifiedNewton")

    for jj in range(500):
        #     print(jj)
        ops.analysis("Transient")
        ok = ops.analyze(1, 0.01)
        # print(ok,jj)

    # ops.printA('-file','A.txt')

    ops.wipeAnalysis

    ops.wipeAnalysis()
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.analysis("Transient")

    # Mass
    ops.integrator("GimmeMCK", 1.0, 0.0, 0.0)
    ops.analyze(1, 0.0)

    # Number of equations in the model
    N = ops.systemSize()  # Has to be done after analyze

    M = ops.printA("-ret")  # Or use ops.printA('-file','M.out')
    M = np.array(M)  # Convert the list to an array
    M.shape = (N, N)  # Make the array an NxN matrix

    # Stiffness
    ops.integrator("GimmeMCK", 0.0, 0.0, 1.0)
    ops.analyze(1, 0.0)
    K = ops.printA("-ret")
    K = np.array(K)
    K.shape = (N, N)

    # Damping
    ops.integrator("GimmeMCK", 0.0, 1.0, 0.0)
    ops.analyze(1, 0.0)
    C = ops.printA("-ret")
    C = np.array(C)
    C.shape = (N, N)

    np.savetxt(f"{data_folder}/M.txt", M)
    np.savetxt(f"{data_folder}/C.txt", C)
    np.savetxt(f"{data_folder}/K.txt", K)
    return Y, K


class Data:
    def __init__(self, device):
        self.device = device
        try:
            self.import_data()
        except:
            get_data()
            self.import_data()

        self.get_fourier_transforms()

    def import_data(self):
        fn = "data"
        if not os.path.exists(fn):
            get_data()
        dirs = os.listdir(fn)
        data = {
            name.split(".")[0]: np.loadtxt(os.path.join(fn, name), max_rows=290)
            for name in dirs
        }

        self.M = data["M"]
        self.C = data["C"]
        self.K = data["K"]
        self.a0, self.a1 = data["Damp_param"]
        self.k_basis = data["k_basis"]
        self.load_magnitude = data["load"] * 1e3
        self.time = data["t"]
        self.u = np.array(
            [
                data["Disp_3_1_2D"],
                data["Disp_3_2D"],
                data["Disp_4_1_2D"],
                data["Disp_4_2D"],
            ]
        )
        self.du_dt = np.array(
            [
                data["Vel_3_1_2D"],
                data["Vel_3_2D"],
                data["Vel_4_1_2D"],
                data["Vel_4_2D"],
            ]
        )
        self.Y = data["Y"]

        self.f = np.zeros_like(self.u)
        self.f[3, :] = -self.load_magnitude

        self.dt = self.time[1] - self.time[0]
        self.n = len(self.time)

        self.TORCH_M = torch.Tensor(self.M).to(self.device)
        self.TORCH_C = torch.Tensor(self.C).to(self.device)
        self.TORCH_K = torch.Tensor(self.K).to(self.device)
        self.TORCH_k_basis = torch.Tensor(self.k_basis).to(self.device)

    def get_fourier_transforms(self):
        self.u_hat = np.fft.fft(self.u)  # Fourier amplitudes
        self.f_hat = np.fft.fft(self.f)  # Fourier amplitudes

        N = self.n
        L = np.ptp(self.time)

        k = self.u_hat[0, :].real * 0
        k[: int(N / 2)] = np.arange(0, N / 2)
        k[int(N / 2 + 1) :] = np.arange(-N / 2 + 1, 0)
        k[int(N / 2)] = 0

        self.xi = k * 2 * np.pi / L  # Nyquist frequency

        self.TORCH_U_HAT = torch.complex(
            torch.Tensor(self.u_hat.real), torch.Tensor(self.u_hat.imag)
        ).to(self.device)
        self.TORCH_F_HAT = torch.complex(
            torch.Tensor(self.f_hat.real), torch.Tensor(self.f_hat.imag)
        ).to(self.device)
        self.TORCH_XI = torch.Tensor(self.xi).to(self.device)

    def get_force_spectral_tensor(self, xi: torch.Tensor) -> torch.Tensor:
        if isinstance(xi, torch.Tensor):
            if xi.requires_grad:
                xi = xi.detach().cpu().numpy()
            else:
                xi = xi.cpu().numpy()

        idx = self.xi.argsort()
        load_hat = np.interp(xi, self.xi[idx], self.f_hat[3, idx])

        f_hat = np.zeros([len(xi), 4], dtype=np.complex128)
        f_hat[:, 3] = load_hat
        return torch.Tensor(f_hat.real).to(self.device), torch.Tensor(f_hat.imag).to(
            self.device
        )


ops_data = Data("cuda")

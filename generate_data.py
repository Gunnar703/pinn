import openseespy.opensees as ops
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_images(data_folder="data"):
    """
    Generate data images for CNN. Opensees docs: https://openseespydoc.readthedocs.io/en/latest/index.html

    Args:
        data_folder (str, optional): folder to save data in. Defaults to "data".
    """

    img_path = os.path.join(data_folder, "images")
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    for fn in os.listdir(img_path):
        os.unlink(os.path.join(img_path, fn))

    RHO = 2 * 10**3
    nu_list = np.linspace(0.2, 0.4, 50)
    Vs_list = np.linspace(100, 400, 50)

    nu_list, Vs_list = (x.reshape(-1, 1) for x in np.meshgrid(nu_list, Vs_list))

    Vp_list = Vs_list * np.sqrt(2 * (1 - nu_list) / (1 - 2 * nu_list))
    G_list = Vs_list**2 * RHO
    Y_list = 2 * G_list / (1 + nu_list)
    _, unique_indices = np.unique(Y_list, return_index=True)

    Vp_list = Vp_list[unique_indices]
    Vs_list = Vs_list[unique_indices]
    nu_list = nu_list[unique_indices]
    G_list = G_list[unique_indices]
    Y_list = Y_list[unique_indices]

    ## Get damping parameters
    omega1 = 2 * np.pi * 1.5  # 1.5 hz first mode
    omega2 = 2 * np.pi * 14
    damp1 = 0.05
    damp2 = 0.1
    a0 = (2 * damp1 * omega1 * (omega2**2) - 2 * damp2 * omega2 * (omega1**2)) / (
        (omega2**2) - (omega1**2)
    )
    a1 = (2 * damp2 * omega2 - 2 * damp1 * omega1) / ((omega2**2) - (omega1**2))

    # Save damping parameters
    with open(os.path.join(data_folder, "Damp_params.txt"), "w") as f:
        np.savetxt(f, np.array([a0, a1]))

    K_list = []
    for Y, nu in zip(Y_list.squeeze(), nu_list.squeeze()):
        u1, u3, M, K = run_ops_once(
            data_folder=data_folder,
            nu=nu,
            rho=RHO,
            elastic_modulus=Y,
            a0=a0,
            a1=a1,
            get_MCK=True,
        )
        image = np.hstack((u1.reshape(-1, 1), u3.reshape(-1, 1)))

        K_list.append(K)

        # Save image to data folder
        with open(
            os.path.join(data_folder, "images", "%s.dat" % str(Y).replace(".", "_")),
            "w",
        ) as f:
            np.savetxt(f, image)

    # Save M for later retrieval
    with open(os.path.join(data_folder, "M.txt"), "w") as f:
        np.savetxt(f, M)

    # Turn K into a tensor and find K_basis
    K_list = np.array(K_list)
    print("K_list:", K_list.shape)
    K_b = np.zeros_like(K_list[0])
    for row in range(K_list.shape[1]):
        for col in range(K_list.shape[2]):
            K_ij = K_list[:, row, col]
            E = Y_list

            K_b[row, col] = np.linalg.lstsq(E, K_ij, rcond=None)[0].squeeze()

    with open(os.path.join(data_folder, "K_basis.txt"), "w") as f:
        np.savetxt(f, K_b)


def run_ops_once(data_folder, nu, rho, elastic_modulus, a0, a1, get_MCK=False):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)

    # Create nodes
    #   4 ----- 3  <- free
    #   |       |
    #   |   E   |
    #   |       |
    #   1 ----- 2  <- fixed

    ops.node(1, 0, 0)  # (element_id, x, y)
    ops.node(2, 5, 0)  # (element_id, x, y)
    ops.node(3, 5, 5)  # (element_id, x, y)
    ops.node(4, 0, 5)  # (element_id, x, y)

    ops.fix(1, 1, 1, 1)  # (element_id, x, y, z): 0 = free, 1 = fixed
    ops.fix(2, 1, 1, 1)  # (element_id, x, y, z): 0 = free, 1 = fixed

    # Create element
    # (material type, material_id, Young's modulus, Poisson's ratio, density)
    ops.nDMaterial("ElasticIsotropic", 100, elastic_modulus, nu, rho)

    # (element_type, element_id, [node_id]*4, thickness, behavior, material_id)
    ops.element("quad", 1000, 1, 2, 3, 4, 1.0, "PlaneStrain", 100)

    # Create Recorders
    ops.recorder(
        "Node",
        "-file",
        os.path.join(data_folder, "u1_tmp.txt"),
        "-node",
        3,
        "-dof",
        2,
        "disp",
    )
    ops.recorder(
        "Node",
        "-file",
        os.path.join(data_folder, "u3_tmp.txt"),
        "-node",
        4,
        "-dof",
        2,
        "disp",
    )

    time_history = os.path.join(data_folder, "t.txt")
    load_history = os.path.join(data_folder, "load.txt")

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

    ops.test("NormDispIncr", 1e-12, 800, 1)
    ops.algorithm("ModifiedNewton")

    for _ in range(500):
        ops.analysis("Transient")
        ops.analyze(1, 0.01)

    ops.wipeAnalysis()
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.analysis("Transient")

    M = K = None
    if get_MCK:
        # Mass
        ops.integrator("GimmeMCK", 1.0, 0.0, 0.0)
        ops.analyze(1, 0.0)

        # Number of equations in the model
        N = ops.systemSize()  # Has to be done after analyze

        M = ops.printA("-ret")  # Or use ops.printA('-file','M.out')
        M = np.array(M)  # Convert the list to an array
        M.reshape(N, N)

        # Stiffness
        ops.integrator("GimmeMCK", 0.0, 0.0, 1.0)
        ops.analyze(1, 0.0)
        K = ops.printA("-ret")
        K = np.array(K)
        K.shape = (N, N)

        # Damping
        # ops.integrator("GimmeMCK", 0.0, 1.0, 0.0)
        # ops.analyze(1, 0.0)
        # C = ops.printA("-ret")
        # C = np.array(C)
        # C.shape = (N, N)

        # Get u history
        with open(os.path.join(data_folder, "u1_tmp.txt")) as f:
            u1 = np.loadtxt(f, max_rows=290)

        with open(os.path.join(data_folder, "u3_tmp.txt")) as f:
            u3 = np.loadtxt(f, max_rows=290)

    return u1, u3, M, K


def main():
    generate_images(data_folder="data")


if __name__ == "__main__":
    main()

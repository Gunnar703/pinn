import openseespy.opensees as ops
import os as os
import numpy as np
import os


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

    ops.recorder(
        "Node", "-file", f"{data_folder}/Vel_3_2D.txt", "-node", 3, "-dof", 2, "vel"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Vel_4_2D.txt", "-node", 4, "-dof", 2, "vel"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Disp_3_2D.txt", "-node", 3, "-dof", 2, "disp"
    )
    ops.recorder(
        "Node", "-file", f"{data_folder}/Disp_4_2D.txt", "-node", 4, "-dof", 2, "disp"
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

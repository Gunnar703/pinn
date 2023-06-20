import numpy as np
import matplotlib.pyplot as plt

loss_arr = np.loadtxt("loss.dat")

steps = loss_arr[:, 0]
physics_loss = loss_arr[:, 1]
max_res_loss = loss_arr[:, 2]
collocation_loss = loss_arr[:, 3:5].sum(axis=1)

plt.figure()
plt.plot(steps, physics_loss, label="Physics Loss", linewidth=1)
plt.plot(
    steps,
    max_res_loss,
    label=r"$\max\left(|residual|\right)$",
    linewidth=1,
)
plt.plot(steps, collocation_loss, label="IC Loss", linewidth=1)
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.yscale("log")
plt.savefig("loss_plot.png")
plt.show()

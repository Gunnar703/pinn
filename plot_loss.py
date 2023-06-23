import numpy as np
import matplotlib.pyplot as plt

loss_arr = np.loadtxt("loss.dat")

steps = loss_arr[:, 0]
physics_loss = loss_arr[:, 1]
collocation_loss = loss_arr[:, 2:4].sum(axis=1)

plt.figure()
plt.plot(steps, physics_loss, label="Physics Loss", linewidth=1)
plt.plot(steps, collocation_loss, label="BC Loss", linewidth=1)
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.yscale("log")
plt.savefig("loss_plot.png")
plt.show()

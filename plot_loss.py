import numpy as np
import matplotlib.pyplot as plt
import os

loss_arr = np.loadtxt("out_files/loss.dat")

steps = loss_arr[:, 0]
physics_loss = loss_arr[:, 1]

velocity_loss = loss_arr[:, 2:6].sum(axis=1)
# position_loss = loss_arr[:, 4:6].sum(axis=1)
# acceleration_loss = loss_arr[:, 6:].sum(axis=1)

if not os.path.isdir("media"):
    os.mkdir("media")

plt.figure()
plt.plot(steps, physics_loss, label="Physics Loss", linewidth=1)
plt.plot(steps, velocity_loss, label="Velocity Loss", linewidth=1)
# plt.plot(steps, position_loss, label="Position Loss", linewidth=1)
# plt.plot(steps, acceleration_loss, label="Acceleration Loss", linewidth=1)
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.yscale("log")
plt.savefig("media/loss_plot.png")
plt.show()

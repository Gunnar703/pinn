import os
import imageio
from matplotlib import pyplot as plt


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def make_training_plot(
    folder_path="plots", output_path="media/training.gif", duration=8
):
    # Get the list of PNG files in the folder
    file_names = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    file_names.sort(
        key=lambda x: int(x[:-4])
    )  # Sort filenames based on the numeric part

    # Create a list to store the image frames
    frames = []

    # Read each image and append it to the frames list
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        image = imageio.imread(file_path)
        frames.append(image)

    # Save the frames as a GIF file
    imageio.mimsave(
        output_path, frames, duration=duration
    )  # Adjust the duration as desired

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        os.unlink(file_path)


def make_loss_plot(loss_history: dict[str:list], output_path="media/loss_plot.png"):
    plt.figure()

    markers = ["8", "s", "*"]
    x = loss_history["epochs"]

    i = 0
    for k, v in loss_history.items():
        if k == "epochs":
            continue
        plt.plot(x, v, label=k, marker=markers[i % len(markers)])
        i += 1

    plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

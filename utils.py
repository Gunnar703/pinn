import os
import imageio


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

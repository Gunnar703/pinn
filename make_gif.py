from PIL import Image
import glob

frames = []
images = glob.glob("plots/train_history/epoch_*_prediction.png")

def sort_by_number(filename):
    number = int(filename.split('_')[2].split('_')[0])  # Extract the number from the file name
    return number

sorted_images = sorted(images, key=sort_by_number)

for image in sorted_images:
    frames.append(Image.open(image))

frames[0].save('training.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=45,
               loop=0)
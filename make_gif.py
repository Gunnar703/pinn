from PIL import Image
import glob

frames = []
images = glob.glob("plots/training/epoch_*_prediction.png")

def sort_by_number(filename):
    number = int(filename.split('_')[1].split('_')[0])  # Extract the number from the file name
    return number

sorted_images = sorted(images, key=sort_by_number)

for image in sorted_images:
    frames.append(Image.open(image))

frames[0].save(
    'training.gif', 
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=100,
    loop=0
)
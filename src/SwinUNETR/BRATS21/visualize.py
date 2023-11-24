import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from colorama import Fore, init

init(autoreset=True)

def print_rainbow_text(text):
    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.MAGENTA]
    rainbow_text = ""
    color_index = 0

    for char in text:
        if char == ' ':
            color_index = (color_index + 1) % len(colors)
            rainbow_text += char
        else:
            rainbow_text += colors[color_index] + char

    return rainbow_text

def print_center(text):
    total_length = 100
    text = f" {text} "
    padded_text = text.center(total_length, '-')
    print("\n", print_rainbow_text(padded_text), "\n")

def print_center_space(text):
    boundary = "-" * 100
    total_length = len(boundary)
    text = f" {text} "
    padded_text = text.center(total_length, ' ')
    print("\n", print_rainbow_text(padded_text), "\n")

data_dir = "../../../data/brats2021"

img_add = os.path.join(data_dir, "TrainingData/BraTS2021_00006/BraTS2021_00006_flair.nii.gz")
label_add = os.path.join(data_dir, "TrainingData/BraTS2021_00006/BraTS2021_00006_seg.nii.gz")
img = nib.load(img_add).get_fdata()
label = nib.load(label_add).get_fdata()
padded_text= f"image shape: {img.shape}, label shape: {label.shape}"
print(print_rainbow_text(padded_text))

plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[:, :, 78], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 78])
plt.show()

import torchvision
import numpy as np
import matplotlib.pyplot as plt


def show(img, figsize=None):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.imshow(img)
    plt.show()


def show_batch(x, nrow=10, padding=0, figsize=None):
    grid = torchvision.utils.make_grid(x, nrow=nrow, padding=padding)
    show(grid, figsize=figsize)

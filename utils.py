import torch
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


def save_checkpoint(
        model, optimizer, lr_scheduler,
        config, epoch, step, p_checkpoint):
    model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'global_step': step,
        'config': config,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }, p_checkpoint)

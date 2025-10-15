# src/utils.py
import matplotlib.pyplot as plt
import numpy as np

def plot_samples(images, labels, cmap="gray", rows=2, cols=5, class_names=None, save_path=None):
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap=cmap)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        title = str(labels[i])
        if class_names:
            title = class_names[labels[i]]
        plt.xlabel(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

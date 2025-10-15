# src/train.py (part 1: load & visualize)
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets

from utils import plot_samples

def load_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    os.makedirs("reports/figures", exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print("Train shape:", x_train.shape, y_train.shape)
    print("Test shape:", x_test.shape, y_test.shape)

    # Visualize samples
    plot_samples(x_train, y_train, save_path="reports/figures/mnist_samples.png")

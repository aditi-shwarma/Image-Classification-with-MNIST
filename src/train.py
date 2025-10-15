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
# src/train.py (part 2: preprocessing)
def preprocess(x_train, x_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # Add channel dimension: (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return x_train, x_test

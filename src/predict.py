# src/predict.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models

def visualize_predictions(model, num_samples=10, save_path="reports/figures/sample_predictions.png"):
    (_, _), (x_test, y_test) = datasets.mnist.load_data()
    x_test_norm = x_test.astype("float32") / 255.0
    x_test_norm = np.expand_dims(x_test_norm, -1)

    preds = model.predict(x_test_norm[:num_samples], verbose=0).argmax(axis=1)

    plt.figure(figsize=(num_samples*1.5, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x_test[i], cmap="gray")
        plt.title(f"T:{y_test[i]}\nP:{preds[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    model = models.load_model("models/mnist_cnn.keras")
    visualize_predictions(model)

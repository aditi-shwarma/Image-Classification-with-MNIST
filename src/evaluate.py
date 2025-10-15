# src/evaluate.py
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import datasets, models

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test

if __name__ == "__main__":
    os.makedirs("reports/figures", exist_ok=True)
    x_test, y_test = load_data()
    model = models.load_model("models/mnist_cnn.keras")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", bbox_inches="tight")
    plt.close()

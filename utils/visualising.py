import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_model_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def draw_confusion_matrix(evaluation_result, label):
    # TP FP
    # FN TN
    matrix = np.round(
        np.array([[evaluation_result[1], evaluation_result[2]], [evaluation_result[4], evaluation_result[3]]])
    )

    plt.figure(figsize=(4, 3))

    ax = sns.heatmap(data=matrix, annot=True, fmt=".0f")
    ax.invert_yaxis()
    ax.invert_xaxis()

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Confusion matrix for {label}")

import matplotlib.pyplot as plt
import itertools
import numpy as np

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Questa funzione stampa una confusion matrix, la normalizzazione può essere applicata tramite il
    parametro `normalize=True`.
    :param cm:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_predictions(all_data, train_predictions, val_predictions, parameter_name=""):
    plt.title(f"Predictions: {parameter_name}")
    plt.plot(all_data, label="Dataset", linestyle="-")
    plt.plot(train_predictions, label="Train predictions", linestyle="-", fillstyle='none')
    plt.plot(val_predictions, label="Validation predictions", linestyle="-", fillstyle='none')
    plt.legend()
    plt.savefig(f'images/dataset_predictions_{parameter_name}.png')
    plt.show()
    plt.close()

import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier


def train_model(data_dict, model_path):
    """
    Trains a RandomForestClassifier model on the given data and saves it to the specified path.

    Args:
    - data_dict (dict): A dictionary containing 'data' and 'labels' keys with corresponding data and labels.
    - model_path (str): The path where the trained model will be saved.

    Returns:
    - y_predict (numpy array): Predictions made by the trained model.
    - y_test (numpy array): True labels for the test data.
    """
    data = np.asarray(data_dict["data"])
    labels = np.asarray(data_dict["labels"])

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    # Train RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Make predictions
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f"{score:.2%} of samples were classified correctly!")

    # Save trained model
    with open(model_path, "wb") as model_file:
        pickle.dump({"model": model}, model_file)

    return y_predict, y_test


def plot_confusion_matrix(y_predict, y_test, label_class, ax):
    """
    Plots the confusion matrix based on predicted and true labels.

    Args:
    - y_predict (numpy array): Predicted labels.
    - y_test (numpy array): True labels.
    - label_class (list): List of class labels.
    - ax (matplotlib axis): Axis for plotting.

    Returns:
    - None
    """
    # Generate classification report and confusion matrix
    report = classification_report(y_test, y_predict, digits=4)
    matrix = confusion_matrix(y_test, y_predict, labels=label_class)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=label_class
    )

    # Display classification report and confusion matrix
    print(f"Classification Report:\n{report}\n")
    display.plot(cmap=plt.cm.Blues, values_format="g", ax=ax)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    data_path = "../data/data.pickle"
    data_dict = pickle.load(open(data_path, "rb"))

    # Start debugging (dimension should be 42)
    indices_to_remove = []
    for i, val in enumerate(data_dict["data"]):
        if len(val) != 42:
            print(f"Dimension: {len(val)}, line: {i}, Class: {data_dict['labels'][i]}")
            indices_to_remove.append(i)

    # Clean data and labels
    for index in sorted(indices_to_remove, reverse=True):
        del data_dict["data"][index]
        del data_dict["labels"][index]

    train_model(data_dict, ".")

# https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from utilities import load_spectral_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    features, labels = load_spectral_data()
    model = GaussianNB()
    model.fit(features, labels)

    expected = np.array(labels)
    predicted_labels = model.predict(features)

    print(classification_report(expected, predicted_labels,
                                labels=np.unique(predicted_labels)))
    confusion_matrix = confusion_matrix(expected, predicted_labels,
                                        labels=np.unique(labels))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                        display_labels=np.unique(labels)).plot()
    cm_display.ax_.set_xticklabels(labels=np.unique(labels),
                                   rotation=30, horizontalalignment='right')
    plt.show()

# https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
# https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def classify_data(features, labels, test_features, test_labels, classifier_function) -> (np.ndarray, np.ndarray):
    model = classifier_function()
    model.fit(features, labels)
    if test_features.size:
        expected = np.array(test_labels)
        predicted_labels = model.predict(test_features)
    else:
        expected = np.array(labels)
        predicted_labels = model.predict(features)
    report = display_results(expected, predicted_labels, str(model).replace('()', ''))
    return expected, predicted_labels, report


def display_results(expected, predicted_labels, confusion_matrix_title='Confusion Matrix'):
    chart_labels = np.unique(expected)
    report = metrics.classification_report(expected, predicted_labels,
                                           labels=np.unique(predicted_labels), output_dict=True)
    print(report)
    confusion_matrix = metrics.confusion_matrix(expected, predicted_labels,
                                                labels=chart_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=chart_labels).plot()
    cm_display.ax_.set_xticklabels(labels=chart_labels,
                                   rotation=30, horizontalalignment='right')
    plt.title(confusion_matrix_title)
    plt.show()
    return report

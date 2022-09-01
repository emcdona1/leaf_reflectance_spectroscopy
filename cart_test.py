# https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utilities import load_spectral_data, display_results


def cart(features, labels, test_features=np.array([]), test_labels=np.array([])) -> (np.ndarray, np.ndarray):
    model = DecisionTreeClassifier()
    model.fit(features, labels)
    if test_features.size:
        expected = np.array(test_labels)
        predicted_labels = model.predict(test_features)
    else:
        expected = np.array(labels)
        predicted_labels = model.predict(features)
    display_results(expected, predicted_labels, 'Decision Tree')
    return expected, predicted_labels

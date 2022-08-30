# https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utilities import load_spectral_data, display_results


def knn(features, labels, test_features=np.array([]), test_labels=np.array([])):
    model = KNeighborsClassifier(weights='distance')
    model.fit(features, labels)
    if test_features.size:
        expected = np.array(test_labels)
        predicted_labels = model.predict(test_features)
    else:
        expected = np.array(labels)
        predicted_labels = model.predict(features)
    display_results(expected, predicted_labels, 'KNN Classifier')


if __name__ == '__main__':
    X, y = load_spectral_data('subgenus')
    knn(X, y)

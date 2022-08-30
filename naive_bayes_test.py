# https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/

import numpy as np
from sklearn.naive_bayes import GaussianNB
from utilities import load_spectral_data, display_results


def naive_bayes(features, labels, test_features=np.array([]), test_labels=np.array([])):
    model = GaussianNB()
    model.fit(features, labels)
    if test_features.size:
        expected = np.array(test_labels)
        predicted_labels = model.predict(test_features)
    else:
        expected = np.array(labels)
        predicted_labels = model.predict(features)
    display_results(expected, predicted_labels, 'Gaussian Naive Bayes')


if __name__ == '__main__':
    X, y = load_spectral_data('subgenus')
    naive_bayes(X, y)

# https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
# https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from utilities import display_results


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


def knn(features, labels, test_features=np.array([]), test_labels=np.array([])) -> (np.ndarray, np.ndarray):
    model = KNeighborsClassifier(weights='distance')
    model.fit(features, labels)
    if test_features.size:
        expected = np.array(test_labels)
        predicted_labels = model.predict(test_features)
    else:
        expected = np.array(labels)
        predicted_labels = model.predict(features)
    display_results(expected, predicted_labels, 'KNN Classifier')
    return expected, predicted_labels


def naive_bayes(features, labels, test_features=np.array([]), test_labels=np.array([])) -> (np.ndarray, np.ndarray):
    model = GaussianNB()
    model.fit(features, labels)
    if test_features.size:
        expected = np.array(test_labels)
        predicted_labels = model.predict(test_features)
    else:
        expected = np.array(labels)
        predicted_labels = model.predict(features)
    display_results(expected, predicted_labels, 'Gaussian Naive Bayes')
    return expected, predicted_labels


def gradient_boost(features, labels, test_features=np.array([]), test_labels=np.array([])) -> (np.ndarray, np.ndarray):
    model = GradientBoostingClassifier()
    model.fit(features, labels)
    if test_features.size:
        expected = np.array(test_labels)
        predicted_labels = model.predict(test_features)
    else:
        expected = np.array(labels)
        predicted_labels = model.predict(features)
    display_results(expected, predicted_labels, 'Gradient Boosting Classifier')
    return expected, predicted_labels

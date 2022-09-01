from utilities import load_spectral_data
from sklearn.model_selection import train_test_split
from cart_test import cart
from naive_bayes_test import naive_bayes
from knn_test import knn
from sklearn import metrics


def main():
    group = input('''What taxonomy group would you like to use?
    - RhRh (all within section Rhododendron, i.e. the 2nd set of characters in the label)
    - Lapponica (all within subsection Lapponica)
    - all (all samples with at least 5 specimens / label)
    >>> ''').lower()
    level = input('''What classification level would you like to use?
    - subgenus ("all" only)
    - subsection ("RhRh" or "all" only)
    - species
    - species group ("Lapponica" only)
    >>> ''').lower()
    features, labels = load_spectral_data(group, level)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

    if input('Would you like to do feature reduction? (Y/n) >>> ').lower() == 'y':
        feature_reduction = input('''What type of feature reduction?
        - PCA
        - ANOVA
        >>> ''')

    algorithm = input('''Which classification algorithm would you like to use?
    - CART (Classification and Regression Tree)
    - bayes (Naive Bayes)
    - KNN (K-nearest Neighbors)
    >>> ''').lower()
    if algorithm == 'cart':
        expected, predicted = cart(X_train, y_train, X_test, y_test)
    elif algorithm == 'bayes':
        expected, predicted = naive_bayes(X_train, y_train, X_test, y_test)
    elif algorithm == 'knn':
        expected, predicted = knn(X_train, y_train, X_test, y_test)
    pass


if __name__ == '__main__':
    main()

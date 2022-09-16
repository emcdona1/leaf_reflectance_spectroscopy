import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import load_spectral_data, cart, knn, naive_bayes, gradient_boost, pca, anova, mutual_info


def main():
    group = int(input('''What taxonomy group would you like to use?
    1. RhRh (all within section Rhododendron, i.e. the 2nd set of characters in the label)
    2. Lapponica (all within subsection Lapponica)
    3. all (all samples with at least 5 specimens / label)
    >>> '''))
    group = ['-', 'rhrh', 'lapponica', 'all'][group]
    level = int(input('''What classification level would you like to use?
    1. subgenus ("all" only)
    2. subsection ("RhRh" or "all" only)
    3. species
    4. species group ("Lapponica" only)
    >>> '''))
    level = ['_', 'subgenus', 'subsection', 'species', 'species group'][level]
    features, labels = load_spectral_data(group, level)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

    if input('Would you like to do feature reduction? (Y/n) >>> ').lower() == 'y':
        feature_reduction = input('''What type of feature reduction?
        1. PCA
        2. ANOVA
        3. Mutual info
        >>> ''').lower()
        if feature_reduction == '1':
            X_train, y_train, X_test, y_test = pca(X_train, y_train, X_test, y_test, n_features=150)
        elif feature_reduction == '2':
            X_train, y_train, X_test, y_test, fr = anova(X_train, y_train, X_test, y_test, n_features=150)
        elif feature_reduction == '3':
            X_train, y_train, X_test, y_test, fr = mutual_info(X_train, y_train, X_test, y_test, n_features=150)
        else:
            print(f'Warning: {feature_reduction} is not a valid selection; all features will be used.')

    algorithm = int(input('''Which classification algorithm would you like to use?
    1. Classification Tree
    2. Naive Bayes
    3. k-Nearest Neighbors
    4. Gradient Boost
    >>> '''))
    algorithm = ['_', 'cart', 'bayes', 'knn', 'gradient'][algorithm]
    if algorithm == 'cart':
        expected, predicted = cart(X_train, y_train, X_test, y_test)
    elif algorithm == 'bayes':
        expected, predicted = naive_bayes(X_train, y_train, X_test, y_test)
    elif algorithm == 'knn':
        expected, predicted = knn(X_train, y_train, X_test, y_test)
    elif algorithm == 'gradient':
        expected, predicted = gradient_boost(X_train, y_train, X_test, y_test)
    outcome = classification_report(expected, predicted, labels=np.unique(predicted), output_dict=True)


if __name__ == '__main__':
    main()

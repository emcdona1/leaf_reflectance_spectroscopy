# https://machinelearningmastery.com/get-your-hands-dirty-with-scikit-learn-now/
# https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/

import pandas as pd
from sklearn.model_selection import train_test_split
from utilities import load_spectral_data, pca, anova, mutual_info, classify_data
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier


def main():
    X_train, X_test, y_train, y_test = load_classes()
    X_test, X_train, y_test, y_train, feature_reduction_name = implement_feature_reduction(X_test, X_train,
                                                                                           y_test, y_train)

    idx = int(input('''Which classification algorithm would you like to use?
    1. Decision Tree
    2. Naive Bayes (Gaussian)
    3. k-Nearest Neighbors
    4. Gradient Boost
    5. Random Forest
    6. Ada-Boost
    >>> '''))
    algorithm = ['_', DecisionTreeClassifier, GaussianNB, KNeighborsClassifier,
                 GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier]
    expected, predicted, classification_report = classify_data(X_train, y_train, X_test, y_test, algorithm[idx])
    print(pd.DataFrame(classification_report).T)


def implement_feature_reduction(X_test, X_train, y_test, y_train):
    name = 'none'
    if input('Would you like to do feature reduction? (Y/n) >>> ').lower() == 'y':
        feature_reduction = input('''What type of feature reduction?
        1. PCA
        2. ANOVA
        3. Mutual info
        >>> ''').lower()
        if feature_reduction == '1':
            X_train, y_train, X_test, y_test = pca(X_train, y_train, X_test, y_test, n_features=150)
            name = 'pca'
        elif feature_reduction == '2':
            X_train, y_train, X_test, y_test, fr = anova(X_train, y_train, X_test, y_test, n_features=150)
            name = 'anova'
        elif feature_reduction == '3':
            X_train, y_train, X_test, y_test, fr = mutual_info(X_train, y_train, X_test, y_test, n_features=150)
            name = 'mutual_info'
        else:
            print(f'Warning: {feature_reduction} is not a valid selection; all features will be used.')
    return X_test, X_train, y_test, y_train, name


def load_classes():
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
    level = ['_', 'subgenus', 'subsection', 'species', 'species_group'][level]
    features, labels = load_spectral_data(group, level)
    return train_test_split(features, labels, test_size=0.2, stratify=labels)


if __name__ == '__main__':
    main()

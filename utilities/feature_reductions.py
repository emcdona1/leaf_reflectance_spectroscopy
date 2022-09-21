import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split


def pca(X_train, y_train, X_test=None, y_test=None, n_features=10) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA):
    if X_test.shape[0] != 0:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2, random_state=1,
                                                            shuffle=True, stratify=y_train)
    p = PCA(n_components=n_features)
    p.fit(X_train)
    X_transformed_train = p.transform(X_train)
    X_transformed_test = p.transform(X_test)
    # _view_pca(X_transformed, y, level)
    return X_transformed_train, y_train, X_transformed_test, y_test  #, p


def _view_pca(X_transformed, labels, level):
    component_1 = 0
    component_2 = 1
    target_names = pd.unique(labels)
    colors = ["navy", "turquoise", "darkorange", "violet", "red"]
    for color, target_name in zip(colors, target_names):
        plt.scatter(
            X_transformed[labels == target_name, component_1],
            X_transformed[labels == target_name, component_2],
            color=color, alpha=0.8, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title(f'{level.capitalize()} PCA #s {component_1} & {component_2}')
    plt.show()


def anova(X_train, y_train, X_test=None, y_test=None, n_features='all') \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, SelectKBest):
    if X_test.shape[0] != 0:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2, random_state=1,
                                                            shuffle=True, stratify=y_train)
    fs = SelectKBest(score_func=f_classif, k=n_features)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)  # note: this doesn't do anything when k='all'
    X_test_fs = fs.transform(X_test)  # note: this doesn't do anything when k='all'
    # _view_feature_usefulness(fs)
    return X_train_fs, y_train, X_test_fs, y_test, fs


def _view_feature_usefulness(fs):
    for i in range(len(fs.scores_)):
        print(f'Feature {fs.feature_names_in_[i]}: {fs.scores_[i]}')
    # plot the scores
    plt.plot([int(i) for i in fs.feature_names_in_], fs.scores_)
    plt.xlabel('nanometers')
    plt.ylabel('ANOVA/f-score')
    plt.show()


def mutual_info(X_train, y_train, X_test=None, y_test=None, n_features='all') \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, SelectKBest):
    if X_test.shape[0] != 0:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                            test_size=0.2, random_state=1,
                                                            shuffle=True, stratify=y_train)
    kbest = SelectKBest(score_func=mutual_info_classif, k=n_features)
    kbest.fit(X_train, y_train)
    X_train_fs = kbest.transform(X_train)  # note: this doesn't do anything when k='all'
    X_test_fs = kbest.transform(X_test)  # note: this doesn't do anything when k='all'
    _view_feature_usefulness(kbest)
    # kbest.feature_names_in_
    return X_train_fs, y_train, X_test_fs, y_test, kbest

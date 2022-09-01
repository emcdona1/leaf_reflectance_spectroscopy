import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


def pca(features, labels, level='species group', n_features=10):
    p = PCA(n_components=n_features)
    component_1 = 0
    component_2 = 1
    target_names = pd.unique(labels)
    colors = ["navy", "turquoise", "darkorange", "violet", "red"]
    features_transformed = p.fit(features).transform(features)
    plt.clf()
    for color, target_name in zip(colors, target_names):
        plt.scatter(
            features_transformed[labels == target_name, component_1],
            features_transformed[labels == target_name, component_2],
            color=color, alpha=0.8, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title(f'{level.capitalize()} PCA #s {component_1} & {component_2}')
    plt.show()
    return p


def anova_reduce_features(X_train, y_train, num_features='all',
                          X_test=None, y_test=None):
    if not X_test:
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15,
                                                            random_state=1, shuffle=True, stratify=y)
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k=num_features)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)  # note: this doesn't do anything because k='all'
    # transform test input data
    X_test_fs = fs.transform(X_test)  # note: this doesn't do anything because k='all'
    # for i in range(len(fs.scores_)):
    #     print(f'Feature {fs.feature_names_in[i]}: {fs.scores_[i]}')
    # plot the scores
    # plt.plot([int(i) for i in fs.feature_names_in_], fs.scores_)
    # plt.title(level.capitalize())
    # plt.xlabel('nanometers')
    # plt.ylabel('ANOVA/f-score')
    # plt.show()
    return (X_train_fs, y_train), (X_test_fs, y_test), fs

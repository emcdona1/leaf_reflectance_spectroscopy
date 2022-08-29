# https://machinelearningmastery.com/feature-selection-with-numerical-input-data/

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from utilities import load_spectral_data


# feature selection
def reduce_features(num_features='all'):
    X, y = load_spectral_data('subgenus', True)
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=1, shuffle=True, stratify=y)
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k=num_features)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)  # note: this doesn't do anything because k='all'
    # transform test input data
    X_test_fs = fs.transform(X_test)  # note: this doesn't do anything because k='all'
    for i in range(len(fs.scores_)):
        print(f'Feature {i}: {fs.scores_[i]}')
    # plot the scores
    plt.plot([int(i) for i in fs.feature_names_in_], fs.scores_, 'ro')
    plt.show()
    return X_train_fs, X_test_fs, fs


def plot_feature_significance():
    plt.clf()
    level = 'subsection'
    X, y = load_spectral_data(level, True)
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X, y)
    # plot the scores
    plt.plot([int(i) for i in fs.feature_names_in_], fs.scores_)
    plt.title(level.capitalize())
    plt.xlabel('nm')
    plt.ylabel('ANOVA/f-score')
    plt.savefig(f'{level}.png')
    plt.show()
    # for i in range(len(fs.scores_)):
    #     print(f'Feature {i}: {fs.scores_[i]}')


def five_fold_validation_of_feature_significance():
    plt.clf()
    level = 'species'
    X, y = load_spectral_data(level, True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(X_train, y_train)
        plt.plot([int(i) for i in fs.feature_names_in_], fs.scores_)
    plt.title(level.capitalize())
    plt.xlabel('nm')
    plt.ylabel('ANOVA/f-score')
    plt.savefig(f'{level}-5_fold.png')
    plt.show()
    # for i in range(len(fs.scores_)):
    #     print(f'Feature {i}: {fs.scores_[i]}')

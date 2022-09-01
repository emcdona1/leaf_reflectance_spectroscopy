# https://machinelearningmastery.com/feature-selection-with-numerical-input-data/

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt


def plot_feature_significance(X, y):
    fs = SelectKBest(score_func=f_classif, k='all')
    fs.fit(X, y)
    # plot the scores
    plt.plot([int(i) for i in fs.feature_names_in_], fs.scores_)
    # plt.title(level.capitalize())
    plt.xlabel('nanometers')
    plt.ylabel('ANOVA/f-score')
    # plt.savefig(f'{level}.png')
    plt.show()
    # for i in range(len(fs.scores_)):
    #     print(f'Feature {fs.feature_names_in[i]}: {fs.scores_[i]}')


def five_fold_validation_of_feature_significance(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(X_train, y_train)
        plt.plot([int(i) for i in fs.feature_names_in_], fs.scores_)
    plt.xlabel('nanometers')
    plt.ylabel('ANOVA/f-score')
    # plt.savefig(f'{level}-5_fold.png')
    plt.show()
    # for i in range(len(fs.scores_)):
    #     print(f'Feature {fs.feature_names_in[i]}: {fs.scores_[i]}')

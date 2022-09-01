import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


if __name__ == '__main__':
    a = 6
    b = 7
    pca(a, b)

import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def load_spectral_data(search_group: str, search_level: str, drop_small=True, search_query=None) -> \
        (pd.DataFrame, pd.Series):
    all_data = _load_data_and_clean()
    all_data = _process_search_group(all_data, search_group)
    all_data = _process_search_level(all_data, search_group, search_level)

    if drop_small:
        classes_to_remove = pd.value_counts(all_data[search_level])
        classes_to_remove = classes_to_remove[classes_to_remove < 5]
        classes_to_remove = list(classes_to_remove.index)
        for remove in classes_to_remove:
            all_data = all_data[all_data[search_level] != remove]

    _, features, _ = np.split(all_data, [2, 423], axis=1)
    labels = all_data[search_level]
    return features, labels


def _process_search_level(all_data, search_group, search_level):
    if search_level == 'subgenus':
        if search_group != 'all':
            raise ValueError(f'Invalid combination: {search_group} and {search_level}')
    elif search_level == 'subsection':
        if search_group == 'lapponica':
            raise ValueError(f'Invalid combination: {search_group} and {search_level}')
        all_data = all_data[all_data['subsection'] != '']
    elif search_level == 'species group':
        if search_group != 'lapponica':
            raise ValueError(f'Invalid combination: {search_group} and {search_level}')
        groups = {'amundsenianum': 2, 'bulu': 4, 'capitatum': 5, 'complexum': 2, 'fastigiatum': 3,
                  'flavidum': 3, 'hippophaeoides': 1, 'impeditum': 3, 'intricatum': 1,
                  'minyaense': 4, 'nitidulum': 1, 'nitidulum_var_omiense': 1, 'nivale': 5,
                  'nivale_ssp_boreale': 5, 'orthocladum': 4, 'orthocladum_var_longistylum': 4,
                  'polycladum': 3, 'rufescens': -1, 'rupicola': 5, 'rupicola_var_muliense': 5,
                  'rupicola_var_rupicola': 5, 'russatum': 5, 'tapeptiforme': 2, 'tapetiforme': 2,
                  'telmateium': 4, 'thymifolium': 1, 'trichanthum': -1, 'tsaii': 1,
                  'websterianum': 1, 'yungningense': 2}
        all_data['species group'] = all_data['species'].apply(lambda f: str(groups[f]))
    elif search_level != 'species':
        raise ValueError(f'Search level value is invalid: {search_level}')
    return all_data


def _process_search_group(all_data, search_group):
    if search_group == 'rhrh':
        all_data = all_data[all_data['subgenus'] == 'Rh']
    elif search_group == 'lapponica':
        all_data = all_data[all_data['subsection'] == 'La']
    elif search_group != 'all':
        raise ValueError(f'Search group value is invalid: {search_group}')
    return all_data


def _load_data_and_clean():
    file_loc = Path('Rhododendron_spectramatrix.csv')
    all_data = pd.read_csv(file_loc, index_col='accession')
    all_data = all_data[~all_data['species'].str.contains('unk')]
    all_data['full_label'] = all_data['species']
    del all_data['species']

    def split_label(a):
        return [''.join(label) for label
                in list(zip(re.findall('[A-Z.]', a), re.split('[A-Z.]', a)[1:]))]

    all_data['genus'] = all_data['full_label'].apply(lambda f: split_label(f)[0])
    all_data['subgenus'] = all_data['full_label'].apply(lambda f: split_label(f)[1])
    all_data['subsection'] = all_data['full_label'].apply(lambda f: split_label(f)[2]
    if len(split_label(f)) == 4
    else '')
    all_data['species'] = all_data['full_label'].apply(lambda f: split_label(f)[-1].replace('.', ''))
    del all_data['full_label']
    return all_data


def display_results(expected, predicted_labels, confusion_matrix_title='Confusion Matrix'):
    plt.clf()
    chart_labels = np.unique(expected)
    print(metrics.classification_report(expected, predicted_labels,
                                        labels=np.unique(predicted_labels)))
    confusion_matrix = metrics.confusion_matrix(expected, predicted_labels,
                                                labels=chart_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=chart_labels).plot()
    cm_display.ax_.set_xticklabels(labels=chart_labels,
                                   rotation=30, horizontalalignment='right')
    plt.title(confusion_matrix_title)
    plt.show()

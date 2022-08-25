import pandas as pd
from pathlib import Path
import re


def load_spectral_data(drop_small=False) -> (pd.DataFrame, pd.Series):
    file_loc = Path('Rhododendron_spectramatrix.csv')
    all_data = pd.read_csv(file_loc, index_col='accession')
    all_data = all_data[~all_data['species'].str.contains('unk')]
    features = all_data.copy().drop('file_names', axis=1) \
        .drop('species', axis=1) \
        .drop('type', axis=1)

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

    label_level_to_use = all_data['subgenus']
    if drop_small:
        classes_to_remove = pd.value_counts(label_level_to_use)
        classes_to_remove = classes_to_remove[classes_to_remove < 5]
        classes_to_remove = list(classes_to_remove.index)
        indicies_to_keep = label_level_to_use[label_level_to_use not in classes_to_remove]
        features = all_data[indicies_to_keep]
        label_level_to_use = label_level_to_use[indicies_to_keep]
    return features, label_level_to_use

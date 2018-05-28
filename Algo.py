import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset.csv")

removed_features = ['objid', 'run', 'rerun', 'fiberid',
                    'camcol', 'specobjid', 'plate', 'mjd',
                    'field']

dataset = dataset.drop(removed_features, axis=1)

corr = dataset.corr()

#sns.heatmap(corr, annot=True)

transformed_features = ['g', 'r', 'i', 'z']

pca_input = dataset[transformed_features]

pca = PCA(n_components=1)
pca_output = pca.fit_transform(X=pca_input)
pca_value = pd.DataFrame(data=pca_output,
                         index=range(len(pca_output)), 
                        columns=['pca_value'])

dataset = dataset.drop(transformed_features, axis=1)

dataset = pd.concat([pca_value, dataset], axis=1)

print(np.unique(dataset['class'], return_counts=True))

stringToIntMap = {'GALAXY':0, 'QSO':1, 'STAR':2}

dataset['class'] = dataset['class'].apply(
        lambda class_value : stringToIntMap[class_value])















import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network
from sklearn import cluster
from sklearn import svm
from sklearn import linear_model

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

print("Labels and their counts")
print(np.unique(dataset['class'], return_counts=True))
print("")

stringToIntMap = {'GALAXY':0, 'QSO':1, 'STAR':2}

dataset['class'] = dataset['class'].apply(
        lambda class_value : stringToIntMap[class_value])

'''
plt.subplot(231)
plt.bar(dataset['redshift'], dataset['class'], width=0.2)
plt.xlabel("Red Shift")
plt.subplot(232)
plt.bar(dataset['pca_value'], dataset['class'], width=0.2)
plt.xlabel("pca_value")
plt.subplot(233)
plt.bar(dataset['ra'], dataset['class'], width=0.2)
plt.xlabel("ra")
plt.subplot(234)
plt.bar(dataset['dec'], dataset['class'], width=0.2)
plt.xlabel("dec")
plt.subplot(235)
plt.bar(dataset['u'], dataset['class'], width=0.2)
plt.xlabel("u")
'''

X = dataset.drop('class', axis=1)
y = dataset['class']

x_train, x_test, y_train, y_test = train_test_split(X, y)



clf = RandomForestClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

f1_score_result = f1_score(y_test, y_pred, average='micro')

print("F1 score for Random Forest Classifier")
print(f1_score_result)
print("")

'''
clf = neural_network.MLPClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

f1_score_result = f1_score(y_test, y_pred, average='micro')

print("F1 score for Neural Network")
print(f1_score_result)
print("")
'''


clf = linear_model.LogisticRegression()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

f1_score_result = f1_score(y_test, y_pred, average='micro')

print("F1 score for Logistic Regression")
print(f1_score_result)
print("")


clf = svm.SVC()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

f1_score_result = f1_score(y_test, y_pred, average='micro')

print("F1 score for SVM")
print(f1_score_result)
print("")
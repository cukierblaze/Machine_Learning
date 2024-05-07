import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

sns.set(font_scale=1.3)
np.random.seed(42)

raw_data=load_iris()
all_data=raw_data.copy()
data=all_data['data']
target=all_data['target']
feature_names=all_data['feature_names']
target_names=all_data['target_names']

df=pd.DataFrame(data=np.c_[data,target],columns=feature_names+['target'])
data=df.iloc[:,[2,1]].values
target=df['target'].apply(int).values

X_train,X_test,y_train,y_test=train_test_split(data,target)

scaler=StandardScaler()
scaler.fit(X_train)
X_test=scaler.transform(X_test)
X_train=scaler.transform(X_train)

classifier=SVC(C=1.0,kernel='linear')
classifier.fit(X_train,y_train)

plt.figure(figsize=(8, 6))
plot_decision_regions(X_train, y_train, classifier)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[1])
plt.title(f'SVC: train accuracy: {classifier.score(X_train, y_train):.4f}')

plt.figure(figsize=(8, 6))
plot_decision_regions(X_test, y_test, classifier)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[1])
plt.title(f'SVC: train accuracy: {classifier.score(X_test, y_test):.4f}')

classifier=SVC(C=1.0,kernel='rbf')
classifier.fit(X_train,y_train)

plt.figure(figsize=(8, 6))
plot_decision_regions(X_train, y_train, classifier)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[1])
plt.title(f'SVC: train accuracy: {classifier.score(X_train, y_train):.4f}')

plt.figure(figsize=(8, 6))
plot_decision_regions(X_test, y_test, classifier)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[1])
plt.title(f'SVC: train accuracy: {classifier.score(X_test, y_test):.4f}')
plt.show()
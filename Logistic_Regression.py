import numpy as np

np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=100, formatter=dict(float=lambda x: f'{x:.2f}'))
np.random.seed(42)

from sklearn.datasets import load_breast_cancer
raw_data = load_breast_cancer()


all_data=raw_data.copy()
data=all_data['data']
target=all_data['target']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(data,target)

from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)
y_prob=log_reg.predict_log_proba(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
from mlxtend.plotting import plot_confusion_matrix
cm=confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm)
print(f'Accuracy={accuracy_score(y_test,y_pred)}')
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

sns.set(font_scale=1.3)
np.random.seed(42)

raw_data=make_moons(n_samples=2000,noise=0.25,random_state=42)
data=raw_data[0]
target=raw_data[1]
df=pd.DataFrame(data=np.c_[data,target],columns=['x1','x2','target'])

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(data,target)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

classifier=DecisionTreeClassifier()
params={
'max_depth':np.arange(1,10),
'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,15,20]
}
grid_search=GridSearchCV(classifier,param_grid=params,scoring='accuracy',cv=5)
grid_search.fit(X_train,y_train)
grid_search.best_params_
from mlxtend.plotting import plot_decision_regions
plt.figure(figsize=(10,8))
plot_decision_regions(X_test,y_test,grid_search)
plt.title(f'Train Set: accuracy: {grid_search.score(X_test, y_test):.4f}')
plt.show()
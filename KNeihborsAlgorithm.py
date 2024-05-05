import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier

sns.set(font_scale=1.3)
np.set_printoptions(precision=6,suppress=True,edgeitems=10,linewidth=1000)
np.random.seed(42)

raw_data=load_iris()
all_data=raw_data.copy()
data=all_data['data']
target=all_data['target']

df=pd.DataFrame(data=np.c_[data,target],columns=all_data['feature_names']+['class'])

_=sns.pairplot(df,vars=all_data['feature_names'],hue='class')

data=data[:,:2]
plt.figure(figsize=(8,6))
plt.scatter(data[:,0],data[:,1],c=target,cmap='viridis')
plt.show()

df = pd.DataFrame(data=np.c_[data,target],columns=['sepal_length','sepal_width','class'])
fig=px.scatter(df,x='sepal_length',y='sepal_width',color='class',width=800)
fig.show()

classifer=KNeighborsClassifier(n_neighbors=5)
classifer.fit(data,target)

x_min,x_max=data[:,0].min()-.5,data[:,0].max()+0.5
y_min,y_max=data[:,1].min()-.5,data[:,1].max()+0.5

xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01), np.arange(y_min,y_max,0.01))
mesh=np.c_(xx.ravel(),yy.ravel())
Z=classifer.predict(mesh)
Z=Z.reshape(xx.shape)

plt.figure(figsize=(10,8))
plt.pcolormesh(xx,yy,Z,cmap='gnuplot',alpha=0.1)
plt.scatter(data[:,0],data[:,1],c=target,cmap='gnuplot',edgecolors='r')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),xx.max())
plt.show()
import numpy as np
import seaborn as sns
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
sns.set(font_scale=1.3)
np.random.seed(42)
np.set_printoptions(precision=6,suppress=True)

data,target=make_regression(n_samples=200,n_features=1,noise=20)
target=target**2

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(data,target)
plot_data=np.arange(-3,3,0.01).reshape(-1,1)
#print((plot_data))
'''
plt.figure(figsize=(8,6))
plt.plot(plot_data, regressor.predict(plot_data), c='green')
plt.scatter(data,target)
plt.show()
'''
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import io
from IPython.display import Image
import pydotplus
regressor=DecisionTreeRegressor(max_depth=2)
regressor.fit(data,target)
plt.figure(figsize=(8,6))
plt.plot(plot_data,regressor.predict(plot_data),c='green')
plt.scatter(data,target)
plt.show()
dot_data=io.StringIO()
export_graphviz(regressor,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=['cecha x'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('graph.png')
Image(graph.create_png(),width=600)

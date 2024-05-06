import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from IPython.display import Image
from io import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier

sns.set(font_scale=1.3)
np.random.seed(42)

raw_data = load_iris()
all_data = raw_data.copy()
data = all_data['data']
target = all_data['target']
feature_names = [name.replace(' ', '_')[:-5] for name in all_data['feature_names']]
target_names = all_data['target_names']

df = pd.DataFrame(data=np.c_[data, target], columns=feature_names + ['target'])
plt.figure(figsize=(8, 6))
_ = sns.scatterplot(x='sepal_length',y= 'sepal_width', hue='target', data=df, legend='full', palette=sns.color_palette()[:3])

data = df.copy()
data = data[['sepal_length', 'sepal_width', 'target']]
target = data.pop('target')
data = data.values
target = target.values.astype('int64')

classifier = DecisionTreeClassifier(max_depth=1, random_state=42)
classifier.fit(data, target)

acc = classifier.score(data, target)
plt.figure(figsize=(8, 6))
plot_decision_regions(data, target, clf=classifier, legend=2)
plt.title(f'Decision Tree: max_depth=1, accuracy: {acc*100:.2f}%')
plt.show()

dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data, feature_names=feature_names[:2], class_names=target_names,
                special_characters=True, rounded=True, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('graph.png')
Image(graph.create_png(), width=300)

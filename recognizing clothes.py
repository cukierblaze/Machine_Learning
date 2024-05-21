#%tensorflow_version 2.x
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets.fashion_mnist import load_data
import plotly.figure_factory as ff
np.set_printoptions(precision=12, suppress=True, linewidth=150)
pd.options.display.float_format = '{:.6f}'.format
sns.set(font_scale=1.3)

(X_train,y_train), (X_test,y_test)=load_data()
plt.imshow(X_train[0],cmap='gray_r')
plt.axis('off')
class_names=['T-shirt','Trousers','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Anknle boot']
plt.figure(figsize=(13,10))
for i in range(1,11):
    plt.subplot(1,10,i)
    plt.axis('off')
    plt.imshow(X_train[i-1],cmap='gray_r')
plt.show()

X_train=X_train/255.
X_test=X_test/255.

X_train=X_train.reshape(60000,28*28)
X_test=X_test.reshape(10000,28*28)

from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
def plot_confusion_matrix(cm):
    cm = cm[::-1]
    cm_df = pd.DataFrame(cm, columns=[f'pred{i}' for i in range(10)], index=[f'true{i}' for i in reversed(range(10))])
    fig = ff.create_annotated_heatmap(z=cm_df.values, x=cm_df.columns.tolist(), y=cm_df.index.tolist(), colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=700, height=500, title='Confusion Matrix (Plotly)', font=dict(size=16))
    fig.show()
plot_confusion_matrix(cm)
print(classification_report(y_test,y_pred,target_names=class_names))
results=pd.DataFrame(data={'y_pred':y_pred,'y_test':y_test})
errors=results['y_pred'] != results['y_test']
errors_idxs=list(errors[errors].index)

plt.figure(figsize=(12,10))
for idx,error_idx in enumerate(errors_idxs[:15]):
    image=X_test[error_idx].reshape(28,28)
    plt.subplot(3,5,idx+1)
    plt.axis('off')
    plt.imshow(image,cmap="Greys")
    plt.title(f'True: {results.loc[error_idx,"y_test"]} Prediction: {results.loc[error_idx,"y_pred"]}')
plt.show()
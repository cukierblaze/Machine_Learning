import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Setting the font scale for seaborn
sns.set(font_scale=1.3)
np.random.seed(42)

# Load the digits dataset
raw_digits = datasets.load_digits()
digits = raw_digits.copy()
images = digits['images']
target = digits['target']

# Visualize the first 6 images and their labels
plt.figure(figsize=(12, 10))
for index, (image, label) in enumerate(list(zip(images, target))[:6]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title(f'Label: {label}')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, target, test_size=0.3, random_state=42)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train and evaluate the SVM classifier with a linear kernel
classifier_linear = SVC(gamma=0.001, kernel='linear')
classifier_linear.fit(X_train, y_train)
print("Linear Kernel Accuracy:", classifier_linear.score(X_test, y_test))

# Train and evaluate the SVM classifier with an RBF kernel
classifier_rbf = SVC(gamma=0.001, kernel='rbf')
classifier_rbf.fit(X_train, y_train)
print("RBF Kernel Accuracy:", classifier_rbf.score(X_test, y_test))

# Make predictions with the RBF kernel classifier
y_pred = classifier_rbf.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot the confusion matrix using seaborn
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap=sns.color_palette("rocket", as_cmap=True))
plt.title('Confusion Matrix (Seaborn)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Define a function to plot the confusion matrix using plotly
def plot_confusion_matrix(cm):
    cm = cm[::-1]
    cm_df = pd.DataFrame(cm, columns=[f'pred{i}' for i in range(10)], index=[f'true{i}' for i in reversed(range(10))])
    fig = ff.create_annotated_heatmap(z=cm_df.values, x=cm_df.columns.tolist(), y=cm_df.index.tolist(), colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=700, height=500, title='Confusion Matrix (Plotly)', font=dict(size=16))
    fig.show()

# Plot the confusion matrix using plotly
plot_confusion_matrix(cm)

results=pd.DataFrame(data={'y_pred':y_pred,'y_test':y_test})
errors=results['y_pred'] != results['y_test']
errors_idxs=list(errors[errors].index)

plt.figure(figsize=(12,10))
for idx,error_idx in enumerate(errors_idxs[:5]):
    image=X_test[error_idx].reshape(8,8)
    plt.subplot(2,4,idx+1)
    plt.axis('off')
    plt.imshow(image,cmap="Greys")
    plt.title(f'True: {results.loc[error_idx,"y_test"]} Prediction: {results.loc[error_idx,"y_pred"]}')
plt.show()
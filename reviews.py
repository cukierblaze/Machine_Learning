import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import subprocess
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Ensuring reproducibility
np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, edgeitems=10, linewidth=1000, formatter=dict(float=lambda x: f'{x:.2f}'))



# Load movie reviews dataset
raw_movie = load_files('movie_reviews')
movie = raw_movie.copy()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(movie['data'], movie['target'], random_state=42)

# Transform text data using TF-IDF
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)  # Use transform instead of fit_transform

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

import plotly.figure_factory as ff

def plot_confusion_matrix(cm):
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=['negative', 'positive'], index=['positive', 'negative'])

    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index),
                                      colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=400, height=400, title='Confusion Matrix', font_size=16)
    fig.show()

plot_confusion_matrix(cm)



print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
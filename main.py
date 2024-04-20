import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess the data
df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')
df = df_raw.drop_duplicates()
df = pd.get_dummies(df, drop_first=True)
data = df.copy()
target = data.pop('charges')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# Train the initial linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("R-squared score on test set:", regressor.score(X_test, y_test))

# Backward elimination using p-values
def backward_elimination(X, y, significance_level=0.05):
    num_vars = X.shape[1]
    predictors = list(X.columns)
    for i in range(num_vars):
        ols = sm.OLS(y, X).fit()
        max_pval = max(ols.pvalues)
        if max_pval > significance_level:
            for j in range(num_vars - i):
                if ols.pvalues[j] == max_pval:
                    X = X.drop(X.columns[j], axis=1)
                    predictors.remove(predictors[j])
    return ols, predictors

# Perform backward elimination
ols_model, selected_predictors = backward_elimination(X_train, y_train)

# Print the summary of the final model
print(ols_model.summary())

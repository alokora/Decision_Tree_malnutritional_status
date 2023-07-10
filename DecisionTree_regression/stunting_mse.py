import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("stunting_r.csv")
encoded_df = pd.get_dummies(df[['ISO3Code','UNRegion', 'WB_Latest', 'age', 'sex', 'mothers_education']])

# Concatenate the encoded features with the target variable
encoded_df = pd.concat([encoded_df, df['stunting_estimate']], axis=1)
df=encoded_df
df = df.replace({True: 1, False: 0})
df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = df

# Split the data into features and target variable
X = data.drop('stunting_estimate', axis=1)
y = data['stunting_estimate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor
regressor = DecisionTreeRegressor(random_state=42)

# Fit the regressor to the training data
regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = regressor.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

#output     Mean Squared Error: 2.244213159146841

#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from scipy.optimize import minimize
from bokeh.plotting import figure, show, output_notebook
from sqlalchemy import create_engine
import sqlite3
import matplotlib.pyplot as plt

# Load training data from CSV files
training_data = pd.read_csv('train.csv')

# Load ideal functions data from CSV
ideal_functions_data = pd.read_csv('ideal.csv')

# Load test data from CSV
test_data = pd.read_csv('test.csv')

# Define a function to calculate the sum of squared deviations
def sum_of_squared_deviations(params, x, y):
    y_pred = params[0] * np.sin(params[1] * x + params[2]) + params[3]
    return np.sum((y - y_pred) ** 2)

# Initialize a list to store chosen ideal functions
chosen_functions = []

# Iterate to choose four functions
for i in range(1, 5):
    # Select training data for the current function
    training_subset = training_data[['x', f'y{i}']]
    x_train, y_train = training_subset['x'], training_subset[f'y{i}']

    # Use scipy minimize to fit the function
    initial_guess = [1, 1, 1, 1]
    result = minimize(sum_of_squared_deviations, initial_guess, args=(x_train, y_train))
    params = result.x

    # Store the chosen function
    chosen_functions.append({'function_number': i, 'params': params})

# Add a ChosenFunction column to your test data
test_data['ChosenFunction'] = 0

# Iterate through the test data to determine the chosen function
for index, row in test_data.iterrows():
    x_test, y_test = row['x'], row['y']
    min_deviation = float('inf')
    chosen_function = 0

    # Determine the chosen function based on minimum deviation
    for i, function in enumerate(chosen_functions):
        params = function['params']
        deviation = sum_of_squared_deviations(params, x_test, y_test)
        if deviation < min_deviation:
            chosen_function = i + 1
            min_deviation = deviation

    test_data.at[index, 'ChosenFunction'] = chosen_function

# Create a scatter plot for data visualization
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c']

# Plot training data
for i in range(1, 5):
    plt.scatter(training_data['x'], training_data[f'y{i}'], c=colors[i - 1], label=f'Training Data Y{i}')

# Plot test data with different colors based on the chosen function
for i in range(1, 5):
    test_subset = test_data[test_data['ChosenFunction'] == i]
    plt.scatter(test_subset['x'], test_subset['y'], c=colors[i - 1], label=f'Test Data Chosen Function {i}')

# Configure the legend
plt.legend(loc='upper right')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot Data Visualization')

# Display the scatter plot
plt.show()

# Import the necessary libraries for the second part
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from sqlalchemy import create_engine

# Function to read the dataset
def read_dataset(dataset):
    data = pd.read_csv(dataset)
    X = data['x'].values
    Y_train = data[['y1', 'y2', 'y3', 'y4']].values
    return X, Y_train

# Function to perform interpolation
def interpolate(train_X, train_Y, ideal_X):
    interpolated_Y = np.zeros((len(ideal_X), train_Y.shape[1]))
    for i in range(train_Y.shape[1]):
        f = interp1d(train_X, train_Y[:, i], kind='linear')
        interpolated_Y[:, i] = f(ideal_X)
    return interpolated_Y

# Function to calculate mean squared error (MSE)
def calculate_mse(actual, predicted):
    if actual.ndim != predicted.ndim:
        raise ValueError("Input arrays must have the same number of dimensions")
    if actual.ndim > 1:
        mse_values = []
        for i in range(actual.shape[1]):
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            mse_values.append(mse)
        return mse_values
    else:
        mse = mean_squared_error(actual, predicted)
        return mse

# Load and preprocess training data
train_dataset = 'train.csv'
train_X, train_Y = read_dataset(train_dataset)

# Load and preprocess ideal data
ideal_dataset = 'ideal.csv'
ideal_data = pd.read_csv(ideal_dataset)
ideal_X = ideal_data['x'].values
ideal_Y = ideal_data[['y1', 'y2', 'y3', 'y4']].values

# Perform interpolation
interpolated_Y = interpolate(train_X, train_Y, ideal_X)

# Load and preprocess test data
test_dataset = 'test.csv'
test_data = pd.read_csv(test_dataset)
test_X = test_data['x'].values
test_Y = np.repeat(test_data['y'].values[:, np.newaxis], 4, axis=1)

# Truncate or pad the test data to match the length of interpolated_Y
test_Y = np.pad(test_Y, ((0, interpolated_Y.shape[0] - len(test_Y)), (0, 0)))

# Calculate MSE for each dependent variable
mse_values = calculate_mse(test_Y, interpolated_Y)

# Reshape mse_values to have two dimensions
mse_values = np.array(mse_values)[:, np.newaxis]

# Find the index of the ideal function with the minimum MSE for each dependent variable
min_mse_indexes = np.argmin(mse_values, axis=1)

# Get the corresponding ideal functions and their names
ideal_functions = ideal_Y[min_mse_indexes]
ideal_function_names = ['y1', 'y2', 'y3', 'y4']

# Create a DataFrame to store the results
result_df = pd.DataFrame({
    'Train Dataset': ideal_function_names,
    'Ideal Function': [ideal_function_names[i] for i in min_mse_indexes],
    'MSE Value': mse_values.flatten()
})

# Display the result DataFrame in the console
print(result_df)

# Create a SQLite database and store the result DataFrame
engine = create_engine('sqlite:///results.db', echo=False)
result_df.to_sql('results', con=engine, if_exists='replace')

# Retrieve data from the SQLite database
conn = sqlite3.connect('results.db')
query = 'SELECT * FROM results'
sql_result_df = pd.read_sql_query(query, conn)
conn.close()

# Display the retrieved data
print("\nData retrieved from the SQLite database:")
print(sql_result_df)



# In[ ]:





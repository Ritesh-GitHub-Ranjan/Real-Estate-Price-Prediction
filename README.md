# Real-Estate-Price-Prediction
This project uses deep learning to predict real estate prices based on property features. The dataset, sourced from Kaggle, includes attributes like bedrooms, bathrooms, and square footage. It involves data cleaning, feature engineering, model training, and evaluation to achieve accurate price predictions.

# IMPORT LIBRARIES AND DATASETS

# Import Necessary Libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset.
house_df = pd.read_csv('realestate_prices.csv', encoding = 'ISO-8859-1')

# Display the DataFrame.
house_df

# Display the first 7 rows of the DataFrame.
house_df.head(7)

# Display the last 7 rows of the DataFrame.
house_df.tail(7)

# Display information about the DataFrame.
house_df.info()

# DATA VISUALIZATION

# Plot a scatterplot of square footage of living area vs. price

sns.scatterplot(x = "sqft_living", y = "price", data = house_df)

# Plot histograms for numerical features
house_df.hist(bins = 20, figsize = (20,20), color = 'b')

# Create a heatmap to visualize correlation between numerical features
f, ax = plt.subplots(figsize=(20, 20))
numeric_df = house_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()  # Compute correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)  # Add correlation coefficients

# Add correlation numbers above the heatmap
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        ax.text(j+0.5, i+0.5, "{:.2f}".format(correlation_matrix.iloc[i, j]),
                ha='center', va='center', color='black')
plt.show()

# Select a subset of features for analysis
house_df_sample = house_df[ ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']   ]

# Display the selected features subset
house_df_sample

## Plot the pairplot for the features contained in "house_df_sample"


# Create a copy of the DataFrame
house_df_sample_copy = house_df_sample.copy()

# Replace infinite values with NaN in the copied DataFrame
house_df_sample_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

# Plot a pairplot for the selected features
sns.pairplot(house_df_sample_copy)

# DATA CLEANING AND FEATURE ENGINEERING

# Select a subset of features for analysis
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

# Display the selected features subset
X = house_df[selected_features]

X

y = house_df['price']

# Display the selected features subset.
y

X.shape

y.shape

# Scale the features using MinMaxScaler to ensure all features are on the same scale
from sklearn.preprocessing import MinMaxScaler

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler and transform the features dataframe
X_scaled = scaler.fit_transform(X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

# Display the scaled features dataframe
X_scaled

# Display the shape of the scaled features dataframe
X_scaled.shape

# Display the maximum values of the scaled features
scaler.data_max_

# Display the minimum values of the scaled features
scaler.data_min_

# Reshape y to ensure compatibility with scaler
y = y.values.reshape(-1,1)

# Scale the target variable (price)
y_scaled = scaler.fit_transform(y)

# Display the scaled target variable
y_scaled

# TRAIN A DEEP LEARNING MODEL WITH LIMITED NUMBER OF FEATURES

# Spliting of data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_scaled,test_size = 0.25)

# Display the shape of the training set
X_train.shape

# Display the shape of the testing set
X_test.shape

# Import necessary libraries for creating the neural network model
import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# Define the neural network model
model = Sequential()
model.add(Dense(100, input_dim = 7, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(200, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# Display the summary of the model
model.summary()

# Compile the model
model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

# Train the model and store the training history
epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2)

# EVALUATING TRAINED DEEP LEARNING MODEL PERFORMANCE 

# Get the keys of the training history
epochs_hist.history.keys()

# Plot the training and validation loss over epochs
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])

# Predict the price using the trained model for a sample data point

# 'bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'sqft_above', 'sqft_basement'
X_test_1 = np.array([[ 4, 3, 1960, 5000, 1, 2000, 3000 ]])

scaler_1 = MinMaxScaler()
X_test_scaled_1 = scaler_1.fit_transform(X_test_1)

y_predict_1 = model.predict(X_test_scaled_1)

y_predict_1 = scaler.inverse_transform(y_predict_1)
y_predict_1

# Predict the price using the trained model for the test set
y_predict = model.predict(X_test)


# Plot the model predictions against the true values
plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')


# Convert the scaled predictions and true values back to their original scale
y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)


# Plot the original true values against the original predicted values
plt.plot(y_test_orig, y_predict_orig, "^", color = 'r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
plt.xlim(0, 5000000)
plt.ylim(0, 3000000)

k = X_test.shape[1]
n = len(X_test)
n

k

# Calculate evaluation metrics for the model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

# Print the evaluation metrics
print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

# TRAINING AND EVALUATIN A DEEP LEARNING MODEL WITH INCREASED NUMBER OF FEATURES (INDEPENDANT VARIABLES)

# Selected features with increased number of independent variables

selected_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 
'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = house_df[selected_features]

from sklearn.preprocessing import MinMaxScaler

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dependent variable (target)
y = house_df['price']

# Reshape the target variable
y = y.values.reshape(-1,1)

# Scale the target variable
y_scaled = scaler.fit_transform(y)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)

import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# Define the model architecture
model = Sequential()
model.add(Dense(10, input_dim = 19, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Train the model
epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, verbose = 1, validation_split = 0.2)

# Plot the loss progress during training
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])

# Predict the price using the trained model
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel("Model Predictions")
plt.ylabel("True Value (ground Truth)")
plt.title('Linear Regression Predictions')
plt.show()

# Transform scaled predictions and true values back to original scale
y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# Calculate evaluation metrics
RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# Conclusion
The deep learning model developed for predicting real estate prices exhibits robust performance, as evidenced by the following evaluation metrics:

- **Root Mean Squared Error (RMSE):** 132,595.728
- **Mean Squared Error (MSE):** 17,581,627,053.155125
- **Mean Absolute Error (MAE):** $ 83,866.89849532754
- **R-squared (R2) Score:** 0.8512497349810544 
- **Adjusted R-squared (Adjusted R2) Score:** 0.8510567676246548

These metrics serve as indicators of the model's effectiveness in capturing the variance present in the target variable (price) and in making accurate predictions. The high values of R-squared and Adjusted R-squared suggest that approximately 85% of the variance in real estate prices can be explained by the model. Additionally, the relatively low values of RMSE, MSE, and MAE indicate that, on average, the model's predictions closely align with the true values, with minimal errors.

In conclusion, the deep learning model demonstrates robust predictive performance in estimating real estate prices, thereby offering valuable insights for decision-making within the real estate market.te market.


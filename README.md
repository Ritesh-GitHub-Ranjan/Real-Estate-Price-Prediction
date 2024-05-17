# Real Estate Price Prediction using Deep Learning

This project utilizes a deep learning approach to predict real estate prices based on various features of properties. The dataset used in this project is sourced from Kaggle and includes multiple attributes of houses such as the number of bedrooms, bathrooms, square footage, and other relevant features. The project involves data cleaning, feature engineering, model training, and evaluation to achieve accurate price predictions.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Data Visualization](#data-visualization)
- [Data Cleaning and Feature Engineering](#data-cleaning-and-feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

The objective of this project is to build a deep learning model that can predict real estate prices based on a variety of property features. By leveraging advanced machine learning techniques, the goal is to improve the accuracy of price predictions, which can be beneficial for stakeholders in the real estate market.

## Dataset

The dataset used in this project is sourced from Kaggle and includes the following features:

- Number of bedrooms
- Number of bathrooms
- Square footage of living space
- Square footage of the lot
- Number of floors
- Waterfront view
- Condition and grade of the house
- Year built and year renovated
- Zip code
- Latitude and longitude
- Additional features of the house

## Project Structure

- **Import Libraries and Datasets**
  - Load necessary libraries and the dataset.
- **Data Visualization**
  - Visualize the data to understand the distribution and relationships between features.
- **Data Cleaning and Feature Engineering**
  - Clean the data and create new features to improve model performance.
- **Model Training**
  - Train a deep learning model using TensorFlow and Keras.
- **Model Evaluation**
  - Evaluate the performance of the trained model using various metrics.
- **Results**
  - Present the results of the model evaluation.
- **Conclusion**
  - Summarize the findings and potential improvements.

## Data Visualization

We start by visualizing the dataset to understand the relationships between different features and the target variable (price).

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x="sqft_living", y="price", data=house_df)
house_df.hist(bins=20, figsize=(20,20), color='b')
f, ax = plt.subplots(figsize=(20, 20))
numeric_df = house_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True)
sns.pairplot(house_df_sample)
```

## Data Cleaning and Feature Engineering

We perform data cleaning and feature engineering to prepare the data for model training.

```python
from sklearn.preprocessing import MinMaxScaler

selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']
X = house_df[selected_features]
y = house_df['price']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)
```

## Model Training

We train a deep learning model using TensorFlow and Keras.

```python
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=7, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)
```

## Model Evaluation

We evaluate the model's performance using various metrics.

```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RMSE = np.sqrt(mean_squared_error(y_test_orig, y_predict_orig))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print(f'RMSE = {RMSE}\nMSE = {MSE}\nMAE = {MAE}\nR2 = {r2}\nAdjusted R2 = {adj_r2}')
```

## Results

The deep learning model demonstrates strong performance, as evidenced by the following evaluation metrics:

- Root Mean Squared Error (RMSE): 132,595.728
- Mean Squared Error (MSE): 17,581,627,053.155125
- Mean Absolute Error (MAE): $ 83,866.89849532754
- R-squared (R2) Score: 0.8512497349810544
- Adjusted R-squared (Adjusted R2) Score: 0.8510567676246548

## Conclusion

The deep learning model demonstrates strong predictive performance in estimating real estate prices, providing valuable insights for decision-making in the real estate market. Further improvements could be made by incorporating additional features and fine-tuning the model architecture.

---

This project is a practical demonstration of applying deep learning techniques to predict real estate prices, showcasing the potential of machine learning in real-world applications.

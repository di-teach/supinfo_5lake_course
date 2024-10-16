# Hands-on: Data Science and Forecasting with Databricks using Machine Learning and Deep Learning Models

This hands-on exercise will guide you through the process of data mining, exploratory data analysis, and implementing forecasting models using **Databricks**, **Apache Spark**, and **Deep Learning** frameworks. You will work with a real-world dataset, perform data preprocessing and feature engineering, and build predictive models using Machine Learning algorithms and Deep Learning architectures like **CNN** and **RNN**.

## Objective

- Set up a **Databricks** workspace for data science tasks.
- Import and explore a real-world dataset.
- Perform data preprocessing and feature engineering.
- Implement machine learning models for forecasting.
- Build deep learning models using CNN and RNN architectures.
- Evaluate and compare the performance of different models.
- Optimize models for better accuracy.

---

## Table of Contents

- [Step 1: Set Up Your Databricks Workspace](#step-1-set-up-your-databricks-workspace)
- [Step 2: Import and Explore the Dataset](#step-2-import-and-explore-the-dataset)
- [Step 3: Data Preprocessing and Feature Engineering](#step-3-data-preprocessing-and-feature-engineering)
- [Step 4: Implement Machine Learning Models for Forecasting](#step-4-implement-machine-learning-models-for-forecasting)
- [Step 5: Build Deep Learning Models (LSTM, CNN)](#step-5-build-deep-learning-models-lstm-cnn)
- [Step 6: Evaluate and Compare Models](#step-6-evaluate-and-compare-models)
- [Step 7: Optimize Model Performance](#step-7-optimize-model-performance)
- [Step 8: Clean Up Resources](#step-8-clean-up-resources)
- [Summary of the Activity](#summary-of-the-activity)
- [Additional Resources](#additional-resources)

---

### **Step 1: Set Up Your Databricks Workspace**

Check that your Databricks workspace is set up (if you have not done so already during a previous exercise).

---

## Step 2: Import and explore the Dataset

**Objective**: Load a real-world dataset into Databricks and perform exploratory data analysis (EDA).

### **a. Select a Dataset**

- **Dataset**: We'll use the **Air Quality dataset** from the UCI Machine Learning Repository, which contains time series data of air pollutant levels.
- **Download Link**: [Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

### **b. Upload Data to Databricks**

- **Upload the Dataset**:
  - In the **Data** tab, click **Add Data** > **Upload File**.
  - Upload the `AirQualityUCI.csv` file.
  - The file will be stored in **DBFS** (Databricks File System) at `/FileStore/tables/AirQualityUCI.csv`.

### **c. Create a New Notebook**

- **Create Notebook**:
  - In the **Workspace** tab, click **Create** > **Notebook**.
  - Name the notebook (e.g., `AirQualityAnalysis`) and select **Python** as the language.
  - Attach it to your cluster.

### **d. Load the Data into a DataFrame**

```python
# Load data into a DataFrame
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", ";") \
    .load("/FileStore/tables/AirQualityUCI.csv")

# Show the first few rows
df.show(5)
```

### **e. Explore the Data**

- **Print Schema**:

```python
df.printSchema()
```

- **Statistical Summary**:

```python
df.describe().show()
```

- **Count Rows and Columns**:

```python
print(f"Total Rows: {df.count()}")
print(f"Total Columns: {len(df.columns)}")
```

- **Display Sample Data**:

```python
display(df.limit(100))
```

**Expected Result**: You have loaded the dataset and performed basic exploratory data analysis to understand the data structure.

---

## Step 3: Data Preprocessing and Feature Engineering

**Objective**: Clean the data, handle missing values, and engineer features suitable for modeling.

### **a. Data Cleaning**

- **Handle Missing Values**:

```python
# Replace missing values identified by -200 with None
from pyspark.sql.functions import when

columns_to_clean = df.columns[2:-2]  # Select pollutant columns
for col_name in columns_to_clean:
    df = df.withColumn(col_name, when(df[col_name] == -200, None).otherwise(df[col_name]))
```

- **Drop Rows with Missing Values**:

```python
df = df.na.drop()
print(f"Total Rows after dropping missing values: {df.count()}")
```

### **b. Feature Engineering**

- **Convert Date and Time Columns**:

```python
from pyspark.sql.functions import concat, to_timestamp

# Combine Date and Time into a single timestamp column
df = df.withColumn('DateTime', to_timestamp(concat(df['Date'], df['Time']), 'dd/MM/yyyyHH.mm.ss'))

# Drop original Date and Time columns
df = df.drop('Date', 'Time')
```

- **Select Relevant Features**:

```python
# Select target variable (e.g., 'CO(GT)') and features
target_column = 'CO(GT)'
feature_columns = [col for col in df.columns if col != target_column and col != 'DateTime']

# Assemble features into a vector
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
df_prepared = assembler.transform(df)
```

### **c. Split Data into Training and Test Sets**

```python
# Split data into training and test sets
train_data, test_data = df_prepared.randomSplit([0.8, 0.2], seed=42)
```

**Expected Result**: The data is cleaned, features are engineered, and the dataset is split into training and testing sets.

---

## Step 4: Implement Machine Learning Models for Forecasting

**Objective**: Build and evaluate machine learning models for forecasting the target variable.

### **a. Linear Regression Model**

```python
from pyspark.ml.regression import LinearRegression

# Initialize Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol=target_column)

# Train the model
lr_model = lr.fit(train_data)

# Make predictions
lr_predictions = lr_model.transform(test_data)

# Evaluate the model
from pyspark.ml.evaluation import RegressionEvaluator

lr_evaluator = RegressionEvaluator(labelCol=target_column, predictionCol='prediction', metricName='rmse')
lr_rmse = lr_evaluator.evaluate(lr_predictions)
print(f"Linear Regression RMSE: {lr_rmse}")
```

### **b. Decision Tree Regressor**

```python
from pyspark.ml.regression import DecisionTreeRegressor

# Initialize Decision Tree Regressor
dt = DecisionTreeRegressor(featuresCol='features', labelCol=target_column)

# Train the model
dt_model = dt.fit(train_data)

# Make predictions
dt_predictions = dt_model.transform(test_data)

# Evaluate the model
dt_evaluator = RegressionEvaluator(labelCol=target_column, predictionCol='prediction', metricName='rmse')
dt_rmse = dt_evaluator.evaluate(dt_predictions)
print(f"Decision Tree RMSE: {dt_rmse}")
```

**Expected Result**: Machine learning models are trained and evaluated, and you have obtained baseline performance metrics.

---

## Step 5: Build Deep Learning Models (LSTM, CNN)

**Objective**: Implement deep learning models for forecasting using LSTM and CNN architectures.

### **a. Prepare Data for Deep Learning Models**

- **Collect Data into Pandas DataFrame**

```python
# Select necessary columns
dl_df = df.select('DateTime', target_column).orderBy('DateTime')

# Convert to Pandas DataFrame
dl_pd_df = dl_df.toPandas()
```

- **Handle Indexing**

```python
# Set DateTime as index
dl_pd_df.set_index('DateTime', inplace=True)
```

### **b. Normalize the Data**

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler(feature_range=(0, 1))
dl_pd_df[target_column] = scaler.fit_transform(dl_pd_df[[target_column]])
```

### **c. Create Sequences for Time Series Forecasting**

```python
# Define sequence length
sequence_length = 10

# Create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

data_values = dl_pd_df[target_column].values
X, y = create_sequences(data_values, sequence_length)
```

- **Split into Training and Testing Sets**

```python
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```

- **Reshape Data for LSTM**

```python
# Reshape for LSTM input (samples, timesteps, features)
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```

### **d. Build and Train LSTM Model**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='tanh', input_shape=(sequence_length, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test))
```

### **e. Build and Train CNN Model**

- **Reshape Data for CNN**

```python
# Reshape for CNN input (samples, timesteps, features)
X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```

- **Define and Train CNN Model**

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# Define the CNN model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(sequence_length, 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(1))
cnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=32, validation_data=(X_test_cnn, y_test))
```

**Expected Result**: Deep learning models are built and trained on the time series data.

---

## Step 6: Evaluate and Compare Models

**Objective**: Assess the performance of machine learning and deep learning models and compare their effectiveness.

### **a. Evaluate LSTM Model**

```python
# Make predictions
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Inverse transform predictions
y_pred_lstm_inv = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
from sklearn.metrics import mean_squared_error

lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_inv))
print(f"LSTM RMSE: {lstm_rmse}")
```

### **b. Evaluate CNN Model**

```python
# Make predictions
y_pred_cnn = cnn_model.predict(X_test_cnn)

# Inverse transform predictions
y_pred_cnn_inv = scaler.inverse_transform(y_pred_cnn.reshape(-1, 1))

# Calculate RMSE
cnn_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_cnn_inv))
print(f"CNN RMSE: {cnn_rmse}")
```

### **c. Compare with Machine Learning Models**

- **Display RMSE Scores**

```python
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Decision Tree RMSE: {dt_rmse}")
print(f"LSTM RMSE: {lstm_rmse}")
print(f"CNN RMSE: {cnn_rmse}")
```

### **d. Visualize Predictions**

```python
import matplotlib.pyplot as plt

# Plot actual vs. predicted values for LSTM
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_lstm_inv, label='LSTM Predicted')
plt.legend()
plt.title('LSTM Model Predictions')
plt.show()

# Plot actual vs. predicted values for CNN
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_cnn_inv, label='CNN Predicted')
plt.legend()
plt.title('CNN Model Predictions')
plt.show()
```

**Expected Result**: You have evaluated the models and can compare their performance based on RMSE and visualizations.

---

## Step 7: Optimize Model Performance

**Objective**: Improve model accuracy through hyperparameter tuning and additional techniques.

### **a. Hyperparameter Tuning for LSTM**

```python
# Modify the LSTM model with additional layers and dropout
from tensorflow.keras.layers import Dropout

lstm_model_opt = Sequential()
lstm_model_opt.add(LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)))
lstm_model_opt.add(Dropout(0.2))
lstm_model_opt.add(LSTM(50))
lstm_model_opt.add(Dropout(0.2))
lstm_model_opt.add(Dense(1))
lstm_model_opt.compile(optimizer='adam', loss='mean_squared_error')

# Train the optimized model
lstm_model_opt.fit(X_train_lstm, y_train, epochs=30, batch_size=32, validation_data=(X_test_lstm, y_test))
```

- **Evaluate the Optimized Model**

```python
# Make predictions
y_pred_lstm_opt = lstm_model_opt.predict(X_test_lstm)

# Inverse transform predictions
y_pred_lstm_opt_inv = scaler.inverse_transform(y_pred_lstm_opt.reshape(-1, 1))

# Calculate RMSE
lstm_opt_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_opt_inv))
print(f"Optimized LSTM RMSE: {lstm_opt_rmse}")
```

### **b. Feature Engineering Enhancements**

- **Add Time-Based Features**

```python
# Extract hour and day from DateTime
dl_pd_df['Hour'] = dl_pd_df.index.hour
dl_pd_df['Day'] = dl_pd_df.index.dayofweek

# Normalize new features
dl_pd_df[['Hour', 'Day']] = scaler.fit_transform(dl_pd_df[['Hour', 'Day']])

# Update data_values with new features
data_values = dl_pd_df[[target_column, 'Hour', 'Day']].values

# Create sequences with additional features
def create_sequences_multivariate(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Target variable is the first column
    return np.array(X), np.array(y)

X_mv, y_mv = create_sequences_multivariate(data_values, sequence_length)

# Split and reshape data
train_size = int(len(X_mv) * 0.8)
X_train_mv, X_test_mv = X_mv[:train_size], X_mv[train_size:]
y_train_mv, y_test_mv = y_mv[:train_size], y_mv[train_size:]

# Reshape data for LSTM
X_train_mv = X_train_mv.reshape((X_train_mv.shape[0], X_train_mv.shape[1], X_train_mv.shape[2]))
X_test_mv = X_test_mv.reshape((X_test_mv.shape[0], X_test_mv.shape[1], X_test_mv.shape[2]))
```

- **Train the Model with New Features**

```python
# Define the LSTM model for multivariate data
lstm_mv_model = Sequential()
lstm_mv_model.add(LSTM(50, activation='tanh', input_shape=(sequence_length, X_train_mv.shape[2])))
lstm_mv_model.add(Dense(1))
lstm_mv_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_mv_model.fit(X_train_mv, y_train_mv, epochs=20, batch_size=32, validation_data=(X_test_mv, y_test_mv))
```

- **Evaluate the Multivariate Model**

```python
# Make predictions
y_pred_lstm_mv = lstm_mv_model.predict(X_test_mv)

# Inverse transform predictions
y_pred_lstm_mv_inv = scaler.inverse_transform(y_pred_lstm_mv.reshape(-1, 1))

# Calculate RMSE
lstm_mv_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_lstm_mv_inv))
print(f"Multivariate LSTM RMSE: {lstm_mv_rmse}")
```

**Expected Result**: Improved model performance through optimization techniques.

---

## Step 8: Clean Up Resources

**Objective**: Release resources and clean up the environment.

- **Stop or Terminate Clusters**:
  - In the **Clusters** tab, click **Terminate** on your cluster if it's no longer needed.

- **Delete Unused Data**:
  - Remove any data stored during the exercise if it's no longer needed.

**Expected Result**: Resources are cleaned up, preventing unnecessary costs.

---

## Summary of the Activity

In this comprehensive hands-on session, you have learned how to:

1. **Set up a Databricks workspace** for data science tasks.
2. **Import and explore a real-world dataset** using Spark DataFrames.
3. **Perform data preprocessing and feature engineering** to prepare data for modeling.
4. **Implement machine learning models** for forecasting and evaluate their performance.
5. **Build deep learning models (LSTM and CNN)** for time series forecasting.
6. **Evaluate and compare models** to determine the most effective approach.
7. **Optimize models** through hyperparameter tuning and feature enhancements.
8. **Clean up resources** to maintain an efficient environment.

This exercise demonstrates how Databricks, Apache Spark, and deep learning frameworks can be leveraged for data science and forecasting tasks, providing practical experience in handling real-world data and implementing advanced models.

---

## Additional Resources

- **Databricks Documentation**:
  - [Data Science & Engineering Overview](https://docs.databricks.com/data-science/index.html)
  - [Deep Learning Overview](https://docs.databricks.com/applications/deep-learning/index.html)
- **TensorFlow Documentation**:
  - [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- **Books**:
  - *"Introduction to Data Mining"* by Pang-Ning Tan, Michael Steinbach, Vipin Kumar.
  - *"Deep Learning"* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
- **Articles and Tutorials**:
  - [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
  - [Forecasting with Deep Learning on Databricks](https://databricks.com/blog/2019/04/30/forecasting-with-deep-learning-on-databricks.html)

Feel free to explore further and apply these techniques to other datasets and problems. Happy data mining and forecasting!

---

**Note**: Ensure that you have the necessary permissions and resources available in your Databricks environment to execute 
the deep learning models, as they may require additional configurations or libraries (e.g., TensorFlow installation). 
If you encounter any issues, refer to the Databricks documentation or consult your system administrator.
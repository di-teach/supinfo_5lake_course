# Hands-on: Model Evaluation and Optimization with Databricks

This hands-on exercise will guide you through the process of evaluating and optimizing machine learning models using **Databricks** and **Apache Spark**. You will work with a real-world dataset, implement various machine learning algorithms, evaluate their performance using appropriate metrics, and apply optimization techniques such as hyperparameter tuning and cross-validation to improve model accuracy and generalization.

## Objective

- Set up a **Databricks** workspace for machine learning tasks.
- Import and explore a real-world dataset.
- Implement machine learning models for classification.
- Evaluate model performance using various evaluation metrics.
- Optimize models through hyperparameter tuning and cross-validation.
- Apply regularization and feature selection techniques.
- Interpret and compare model results.
- Understand best practices for model evaluation and optimization.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Set Up Your Databricks Workspace](#step-1-set-up-your-databricks-workspace)
- [Step 2: Import and Explore the Dataset](#step-2-import-and-explore-the-dataset)
- [Step 3: Implement Baseline Machine Learning Models](#step-3-implement-baseline-machine-learning-models)
- [Step 4: Evaluate Model Performance](#step-4-evaluate-model-performance)
- [Step 5: Optimize Models with Hyperparameter Tuning](#step-5-optimize-models-with-hyperparameter-tuning)
- [Step 6: Apply Cross-Validation](#step-6-apply-cross-validation)
- [Step 7: Feature Selection and Regularization](#step-7-feature-selection-and-regularization)
- [Step 8: Compare and Interpret Model Results](#step-8-compare-and-interpret-model-results)
- [Step 9: Clean Up Resources](#step-9-clean-up-resources)
- [Summary of the Activity](#summary-of-the-activity)
- [Additional Resources](#additional-resources)

### **Step 1: Set Up Your Databricks Workspace**

Check that your Databricks workspace is set up (if you have not done so already during a previous exercise).

---

## Step 2: Import and Explore the Dataset

**Objective**: Load a real-world dataset into Databricks and perform exploratory data analysis (EDA).

### **a. Select a Dataset**

- **Dataset**: We'll use the **Bank Marketing dataset** from the UCI Machine Learning Repository, which is commonly used for classification tasks.
- **Download Link**: [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

### **b. Upload Data to Databricks**

- **Upload the Dataset**:
  - In the **Data** tab, click **Add Data** > **Upload File**.
  - Upload the `bank-additional-full.csv` file.
  - The file will be stored in **DBFS** (Databricks File System) at `/FileStore/tables/bank-additional-full.csv`.

### **c. Create a New Notebook**

- **Create Notebook**:
  - In the **Workspace** tab, click **Create** > **Notebook**.
  - Name the notebook (e.g., `BankMarketingEvaluation`) and select **Python** as the language.
  - Attach it to your cluster.

### **d. Load the Data into a DataFrame**

```python
# Load data into a DataFrame
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", ";") \
    .load("/FileStore/tables/bank-additional-full.csv")

# Show the first few rows
df.show(5)
```

### **e. Explore the Data**

- **Print Schema**:

```python
df.printSchema()
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

- **Check Class Distribution**:

```python
df.groupBy('y').count().show()
```

**Expected Result**: You have loaded the dataset and performed basic exploratory data analysis to understand the data structure and class distribution.

---

## Step 3: Implement Baseline Machine Learning Models

**Objective**: Implement baseline classification models to establish a performance benchmark.

### **a. Data Preprocessing**

- **Handle Categorical Variables**:

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# List of categorical columns
categorical_cols = [field for (field, dtype) in df.dtypes if dtype == "string" and field != "y"]

# Index and encode categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column+"_Index") for column in categorical_cols]
encoders = [OneHotEncoder(inputCol=column+"_Index", outputCol=column+"_Vec") for column in categorical_cols]

# Index label column
label_indexer = StringIndexer(inputCol="y", outputCol="label")

# Assemble features
feature_cols = [column+"_Vec" for column in categorical_cols] + [field for (field, dtype) in df.dtypes if dtype != "string" and field != "y"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Build preprocessing pipeline
pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler])

# Fit and transform data
df_prepared = pipeline.fit(df).transform(df)
```

### **b. Split Data into Training and Test Sets**

```python
# Split data into training and test sets
train_data, test_data = df_prepared.randomSplit([0.8, 0.2], seed=42)
```

### **c. Implement a Logistic Regression Model**

```python
from pyspark.ml.classification import LogisticRegression

# Initialize Logistic Regression model
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

# Train the model
lr_model = lr.fit(train_data)

# Make predictions
lr_predictions = lr_model.transform(test_data)
```

### **d. Implement a Decision Tree Classifier**

```python
from pyspark.ml.classification import DecisionTreeClassifier

# Initialize Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')

# Train the model
dt_model = dt.fit(train_data)

# Make predictions
dt_predictions = dt_model.transform(test_data)
```

**Expected Result**: Baseline classification models are trained, and predictions are made on the test dataset.

---

## Step 4: Evaluate Model Performance

**Objective**: Evaluate the performance of the baseline models using appropriate evaluation metrics.

### **a. Evaluate Logistic Regression Model**

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Binary Classification Evaluator
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Calculate AUC
lr_auc = binary_evaluator.evaluate(lr_predictions)
print(f"Logistic Regression AUC: {lr_auc}")

# Multiclass Classification Evaluator
multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Calculate Accuracy
lr_accuracy = multi_evaluator.evaluate(lr_predictions, {multi_evaluator.metricName: "accuracy"})
print(f"Logistic Regression Accuracy: {lr_accuracy}")
```

### **b. Evaluate Decision Tree Model**

```python
# Calculate AUC
dt_auc = binary_evaluator.evaluate(dt_predictions)
print(f"Decision Tree AUC: {dt_auc}")

# Calculate Accuracy
dt_accuracy = multi_evaluator.evaluate(dt_predictions, {multi_evaluator.metricName: "accuracy"})
print(f"Decision Tree Accuracy: {dt_accuracy}")
```

### **c. Confusion Matrix**

```python
# Confusion Matrix for Logistic Regression
lr_predictions.groupBy('label', 'prediction').count().show()

# Confusion Matrix for Decision Tree
dt_predictions.groupBy('label', 'prediction').count().show()
```

**Expected Result**: You have calculated evaluation metrics such as AUC and accuracy for both models and have confusion matrices to understand their performance.

---

## Step 5: Optimize Models with Hyperparameter Tuning

**Objective**: Improve model performance by tuning hyperparameters using grid search.

### **a. Hyperparameter Tuning for Logistic Regression**

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create parameter grid
paramGrid_lr = (ParamGridBuilder()
                .addGrid(lr.regParam, [0.01, 0.1, 1.0])
                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                .build())

# Cross-validator setup
crossval_lr = CrossValidator(estimator=lr,
                             estimatorParamMaps=paramGrid_lr,
                             evaluator=binary_evaluator,
                             numFolds=5)

# Run cross-validation
cv_lr_model = crossval_lr.fit(train_data)

# Make predictions on test data
cv_lr_predictions = cv_lr_model.transform(test_data)

# Evaluate optimized model
cv_lr_auc = binary_evaluator.evaluate(cv_lr_predictions)
print(f"Optimized Logistic Regression AUC: {cv_lr_auc}")
```

### **b. Hyperparameter Tuning for Decision Tree**

```python
# Create parameter grid
paramGrid_dt = (ParamGridBuilder()
                .addGrid(dt.maxDepth, [5, 10, 15])
                .addGrid(dt.maxBins, [32, 64])
                .build())

# Cross-validator setup
crossval_dt = CrossValidator(estimator=dt,
                             estimatorParamMaps=paramGrid_dt,
                             evaluator=binary_evaluator,
                             numFolds=5)

# Run cross-validation
cv_dt_model = crossval_dt.fit(train_data)

# Make predictions on test data
cv_dt_predictions = cv_dt_model.transform(test_data)

# Evaluate optimized model
cv_dt_auc = binary_evaluator.evaluate(cv_dt_predictions)
print(f"Optimized Decision Tree AUC: {cv_dt_auc}")
```

**Expected Result**: You have optimized both models using hyperparameter tuning and observed improvements in their performance metrics.

---

## Step 6: Apply Cross-Validation

**Objective**: Ensure that model evaluation is robust by using cross-validation techniques.

- **Cross-validation was applied in the previous step using `CrossValidator` with `numFolds=5`.**

**Discussion**:

- Cross-validation helps to reduce the variability in the model performance estimation and provides a more reliable assessment.
- The use of `CrossValidator` in Spark MLlib automatically splits the training data into folds and averages the evaluation metrics.

---

## Step 7: Feature Selection and Regularization

**Objective**: Apply feature selection techniques and regularization to improve model generalization.

### **a. Feature Importance from Decision Tree**

```python
# Get feature importances
import pandas as pd

# Extract feature names
input_features = assembler.getInputCols()

# Get feature importances
dt_feature_importances = cv_dt_model.bestModel.featureImportances.toArray()

# Create a DataFrame
feature_importance_df = pd.DataFrame({'feature': input_features, 'importance': dt_feature_importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print(feature_importance_df)
```

### **b. Select Top Features**

```python
# Select top N features (e.g., top 10)
top_features = feature_importance_df['feature'].head(10).tolist()

# Re-assemble features with top features only
assembler_top = VectorAssembler(inputCols=top_features, outputCol="features_top")

# Update the pipeline and re-fit the model
pipeline_top = Pipeline(stages=indexers + encoders + [label_indexer, assembler_top])
df_prepared_top = pipeline_top.fit(df).transform(df)

# Split data
train_data_top, test_data_top = df_prepared_top.randomSplit([0.8, 0.2], seed=42)

# Re-train the model with top features
cv_lr_model_top = crossval_lr.fit(train_data_top)
cv_lr_predictions_top = cv_lr_model_top.transform(test_data_top)
cv_lr_auc_top = binary_evaluator.evaluate(cv_lr_predictions_top)
print(f"Optimized Logistic Regression AUC with Top Features: {cv_lr_auc_top}")
```

### **c. Apply Regularization**

- **Regularization was applied in the Logistic Regression hyperparameter tuning with `regParam` and `elasticNetParam`.**

**Expected Result**: By selecting the most important features and applying regularization, you may observe further improvements in model performance or generalization.

---

## Step 8: Compare and Interpret Model Results

**Objective**: Interpret the evaluation metrics and understand the trade-offs between different models and optimization techniques.

### **a. Compile Results**

```python
results = pd.DataFrame({
    'Model': ['Baseline LR', 'Optimized LR', 'Optimized LR with Top Features', 'Baseline DT', 'Optimized DT'],
    'AUC': [lr_auc, cv_lr_auc, cv_lr_auc_top, dt_auc, cv_dt_auc]
})

print(results)
```

### **b. Interpret Findings**

- **Discuss the following**:
  - Which model performed the best based on AUC and accuracy?
  - How did hyperparameter tuning affect the performance?
  - What impact did feature selection have on the model?
  - Are there any trade-offs between model complexity and performance?
  - How can these models be further improved?

### **c. Visualize ROC Curves**

```python
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def plot_roc(model_predictions, label):
    # Extract ROC data
    roc = model_predictions.select("label", "probability").rdd.map(lambda row: (float(row['probability'][1]), float(row['label']))).collect()
    roc_df = pd.DataFrame(roc, columns=['probability', 'label'])
    roc_df = roc_df.sort_values('probability', ascending=False)
    tpr = []
    fpr = []
    thresholds = roc_df['probability'].unique()
    P = roc_df['label'].sum()
    N = roc_df.shape[0] - P
    for threshold in thresholds:
        TP = roc_df[(roc_df['probability'] >= threshold) & (roc_df['label'] == 1)].shape[0]
        FP = roc_df[(roc_df['probability'] >= threshold) & (roc_df['label'] == 0)].shape[0]
        tpr.append(TP / P)
        fpr.append(FP / N)
    plt.plot(fpr, tpr, label=label)

# Plot ROC curves
plt.figure(figsize=(8,6))
plot_roc(lr_predictions, 'Baseline LR')
plot_roc(cv_lr_predictions, 'Optimized LR')
plot_roc(cv_lr_predictions_top, 'Optimized LR with Top Features')
plot_roc(dt_predictions, 'Baseline DT')
plot_roc(cv_dt_predictions, 'Optimized DT')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
```

**Expected Result**: You have compiled the results, visualized ROC curves, and interpreted the performance of different models.

---

## Step 9: Clean Up Resources

**Objective**: Release resources and clean up the environment.

- **Stop or Terminate Clusters**:
  - In the **Clusters** tab, click **Terminate** on your cluster if it's no longer needed.

- **Delete Unused Data**:
  - Remove any data stored during the exercise if it's no longer needed.

**Expected Result**: Resources are cleaned up, preventing unnecessary costs.

---

## Summary of the Activity

In this comprehensive hands-on session, you have learned how to:

1. **Set up a Databricks workspace** for machine learning tasks.
2. **Import and explore a real-world dataset** using Spark DataFrames.
3. **Implement baseline machine learning models** for classification.
4. **Evaluate model performance** using appropriate evaluation metrics such as AUC, accuracy, and confusion matrices.
5. **Optimize models** through hyperparameter tuning and cross-validation.
6. **Apply feature selection and regularization** to improve model generalization.
7. **Compare and interpret model results**, understanding the trade-offs between different models and techniques.
8. **Clean up resources** to maintain an efficient environment.

This exercise demonstrates how Databricks and Apache Spark can be leveraged for model evaluation and optimization tasks, providing practical experience in improving machine learning models and ensuring they generalize well to unseen data.

---

## Additional Resources

- **Databricks Documentation**:
  - [Machine Learning with Apache Spark MLlib](https://docs.databricks.com/machine-learning/index.html)
  - [Cross-Validation and Hyperparameter Tuning](https://spark.apache.org/docs/latest/ml-tuning.html)
- **Books**:
  - *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"* by Aurélien Géron.
  - *"Spark: The Definitive Guide"* by Bill Chambers and Matei Zaharia.
- **Articles and Tutorials**:
  - [Understanding AUC - ROC Curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
  - [Hyperparameter Tuning the Random Forest in Python](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)
  - [An Introduction to Model Evaluation & Validation](https://towardsdatascience.com/an-introduction-to-model-evaluation-validation-4b4eb9ec3e8a)

Feel free to explore further and apply these techniques to other datasets and problems. Happy modeling and optimizing!

---

**Note**: Ensure that you have the necessary permissions and resources available in your Databricks environment to execute the models, as they may require additional configurations or libraries. If you encounter any issues, refer to the Databricks documentation or consult your system administrator.

---
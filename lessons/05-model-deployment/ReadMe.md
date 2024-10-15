# Hands-on: Deploying and Monitoring Machine Learning Models with MLOps

This hands-on exercise will guide you through the process of deploying and monitoring a machine learning model using MLOps practices with **Databricks** and **MLflow**. You will work with a real-world dataset, train a model, deploy it using MLflow Model Serving, and set up monitoring to track its performance over time.

## Objective

- Set up a **Databricks** workspace and **MLflow** for MLOps tasks.
- Train a machine learning model and track experiments using MLflow.
- Deploy the trained model using MLflow Model Serving.
- Implement model versioning and manage model lifecycle stages.
- Set up monitoring to track model performance and detect drift.
- Configure alerts for performance degradation.
- Automate retraining and deployment.
- Understand best practices for MLOps in real-world scenarios.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Set Up Your Databricks Workspace and MLflow](#step-1-set-up-your-databricks-workspace-and-mlflow)
- [Step 2: Import and Explore the Dataset](#step-2-import-and-explore-the-dataset)
- [Step 3: Train a Machine Learning Model](#step-3-train-a-machine-learning-model)
- [Step 4: Track Experiments with MLflow](#step-4-track-experiments-with-mlflow)
- [Step 5: Register and Manage the Model](#step-5-register-and-manage-the-model)
- [Step 6: Deploy the Model Using MLflow Model Serving](#step-6-deploy-the-model-using-mlflow-model-serving)
- [Step 7: Set Up Model Monitoring](#step-7-set-up-model-monitoring)
- [Step 8: Configure Alerts for Performance Degradation](#step-8-configure-alerts-for-performance-degradation)
- [Step 9: Automate Retraining and Deployment](#step-9-automate-retraining-and-deployment)
- [Step 10: Clean Up Resources](#step-10-clean-up-resources)
- [Summary of the Activity](#summary-of-the-activity)
- [Additional Resources](#additional-resources)

---

## Prerequisites

- Basic knowledge of Python programming.
- Familiarity with machine learning concepts.
- Understanding of model deployment and monitoring.
- Access to a Databricks workspace with MLflow capabilities.

---

### **Step 1: Set Up Your Databricks Workspace**

Check that your Databricks workspace is set up (if you have not done so already during a previous exercise).

---

## Step 2: Import and Explore the Dataset

**Objective**: Load a real-world dataset into Databricks and perform exploratory data analysis (EDA).

### **a. Select a Dataset**

- **Dataset**: We'll use the **Wine Quality dataset** from the UCI Machine Learning Repository, suitable for classification tasks.
- **Download Link**: [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)

### **b. Upload Data to Databricks**

- **Upload the Dataset**:
  - In the **Data** tab, click **Add Data** > **Upload File**.
  - Upload the `winequality-red.csv` file.
  - The file will be stored in **DBFS** (Databricks File System) at `/FileStore/tables/winequality-red.csv`.

### **c. Create a New Notebook**

- **Create Notebook**:
  - In the **Workspace** tab, click **Create** > **Notebook**.
  - Name the notebook (e.g., `WineQualityMLOps`) and select **Python** as the language.
  - Attach it to your cluster.

### **d. Load the Data into a DataFrame**

```python
# Load data into a DataFrame
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", ";") \
    .load("/FileStore/tables/winequality-red.csv")

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

- **Check Class Distribution**:

```python
df.groupBy('quality').count().show()
```

**Expected Result**: You have loaded the dataset and performed basic exploratory data analysis to understand the data structure and class distribution.

---

## Step 3: Train a Machine Learning Model

**Objective**: Train a classification model and prepare it for deployment.

### **a. Data Preprocessing**

- **Label Binarization**:
  - Convert the `quality` variable into a binary classification problem (e.g., good vs. bad wine).

```python
from pyspark.sql.functions import when

# Create a new binary label column
df = df.withColumn('label', when(df['quality'] >= 7, 1).otherwise(0))
```

- **Assemble Features**:

```python
from pyspark.ml.feature import VectorAssembler

feature_cols = df.columns[:-2]  # Exclude 'quality' and 'label'
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_prepared = assembler.transform(df)
```

### **b. Split Data into Training and Test Sets**

```python
# Split data into training and test sets
train_data, test_data = df_prepared.randomSplit([0.8, 0.2], seed=42)
```

### **c. Train a Random Forest Classifier**

```python
from pyspark.ml.classification import RandomForestClassifier

# Initialize Random Forest Classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100)

# Train the model
rf_model = rf.fit(train_data)
```

---

## Step 4: Track Experiments with MLflow

**Objective**: Use MLflow to track the experiment, logging parameters, metrics, and the model.

### **a. Set Up MLflow Tracking**

```python
import mlflow
import mlflow.spark

# Set the experiment (use your own email or path)
mlflow.set_experiment("/Users/your_email@example.com/WineQualityExperiment")
```

### **b. Train the Model with MLflow Tracking**

```python
with mlflow.start_run():
    # Train the model
    rf_model = rf.fit(train_data)

    # Log parameters
    mlflow.log_param("num_trees", rf.getNumTrees())
    mlflow.log_param("max_depth", rf.getMaxDepth())

    # Make predictions
    predictions = rf_model.transform(test_data)

    # Evaluate the model
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    print(f"AUC: {auc}")

    # Log metrics
    mlflow.log_metric("auc", auc)

    # Log the model
    mlflow.spark.log_model(rf_model, "random_forest_model")
```

### **c. View Experiment in MLflow UI**

- Navigate to the **Experiments** tab in Databricks.
- Find your experiment and review the run details.
- **Check**: Parameters, metrics, and model artifacts are logged.

**Expected Result**: The experiment is tracked in MLflow, and you can view all relevant information in the UI.

---

## Step 5: Register and Manage the Model

**Objective**: Register the trained model in the MLflow Model Registry and manage its lifecycle stages.

### **a. Register the Model**

```python
# Register the model
run_id = mlflow.active_run().info.run_id
model_uri = f"runs:/{run_id}/random_forest_model"

model_details = mlflow.register_model(model_uri=model_uri, name="WineQualityRFModel")
```

### **b. Manage Model Versions and Stages**

- **Navigate** to the **Models** tab in Databricks.
- **Find** your registered model **WineQualityRFModel**.
- **Transition** the model version to **Staging** or **Production** as appropriate.

### **c. Add Model Description and Annotations**

- **Document** the model's purpose, training data, and evaluation metrics.
- **Set Tags** for better organization.

**Expected Result**: The model is registered, and you can manage its versions and lifecycle stages.

---

## Step 6: Deploy the Model Using MLflow Model Serving

**Objective**: Deploy the registered model for serving predictions.

### **a. Enable Model Serving in Databricks**

- **In the Models tab**:
  - Select your model **WineQualityRFModel**.
  - Go to the **Serving** tab.
  - Click **Enable Serving**.

### **b. Test the Deployed Model**

- **Get the Serving Endpoint URL**.
- **Send a Test Request**:

```python
import requests
import json

# Replace with your serving endpoint URL
url = "https://<databricks-instance>/model/WineQualityRFModel/Production/invocations"

# Prepare input data
input_data = test_data.limit(1).toPandas()[feature_cols].values.tolist()
input_json = json.dumps({
    "dataframe_records": input_data
})

# Set headers (include authentication if necessary)
headers = {'Content-Type': 'application/json'}

# Make the request
response = requests.post(url, headers=headers, data=input_json)

# Print the response
print(response.json())
```

### **c. Integrate with Applications**

- **Use the REST API** to integrate the model into web or mobile applications.

**Expected Result**: The model is deployed and can serve predictions via REST API.

---

## Step 7: Set Up Model Monitoring

**Objective**: Monitor the model's performance over time to detect drift and degradation.

### **a. Log Predictions and Inputs**

```python
with mlflow.start_run(run_id=run_id):
    # Simulate new data
    new_data = test_data.limit(100)

    # Make predictions
    predictions = rf_model.transform(new_data)

    # Extract true labels and predictions
    y_true = [row['label'] for row in predictions.select('label').collect()]
    y_pred = [row['prediction'] for row in predictions.select('prediction').collect()]

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Log metrics
    mlflow.log_metric("inference_accuracy", accuracy)
    mlflow.log_metric("inference_precision", precision)
    mlflow.log_metric("inference_recall", recall)
```

### **b. Set Up Dashboards**

- **Use MLflow UI** or integrate with tools like **Grafana** to visualize metrics over time.
- **Plot metrics** to observe trends.

**Expected Result**: Model performance metrics during inference are logged and can be monitored over time.

---

## Step 8: Configure Alerts for Performance Degradation

**Objective**: Set up alerts to notify when model performance drops below acceptable thresholds.

### **a. Define Performance Thresholds**

- **Set acceptable ranges** for metrics like accuracy, precision, recall (e.g., accuracy > 0.8).

### **b. Implement Alerting Mechanism**

- **Option 1**: Use Databricks Jobs and Notifications.

  - Create a **Job** that runs a notebook to check the latest metrics.
  - If metrics fall below thresholds, **send an alert** via email or messaging service.

- **Option 2**: Integrate with Monitoring Tools

  - Use tools like **Prometheus** and **Grafana** to set up alerts.

### **c. Example Alert Implementation**

```python
if accuracy < 0.8:
    # Send alert (example using email)
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(f"Alert: Model accuracy dropped to {accuracy}")
    msg['Subject'] = 'Model Performance Alert'
    msg['From'] = 'your_email@example.com'
    msg['To'] = 'team_email@example.com'

    # Send the message via SMTP server
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()
```

**Note**: Replace email addresses and SMTP server details with your actual configuration.

**Expected Result**: Alerts are configured to notify the team when model performance degrades.

---

## Step 9: Automate Retraining and Deployment

**Objective**: Set up automation for retraining the model when necessary.

### **a. Create a Retraining Pipeline**

- **Write a Notebook or Script** that:

  - Loads new training data.
  - Retrains the model.
  - Evaluates the model.
  - Registers and transitions the model to production if it meets performance criteria.

### **b. Schedule the Retraining Job**

- **Use Databricks Jobs** to schedule the retraining pipeline to run periodically or trigger based on events.

### **c. Example Retraining Job**

```python
def retrain_model():
    # Load updated data (e.g., new_data)
    updated_data = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("delimiter", ";") \
        .load("/FileStore/tables/new_winequality_data.csv")
    
    # Preprocess data (same as before)
    # ...

    # Split data
    train_data, test_data = updated_data.randomSplit([0.8, 0.2], seed=42)
    
    # Train model
    new_rf_model = rf.fit(train_data)
    
    # Evaluate model
    new_predictions = new_rf_model.transform(test_data)
    new_auc = evaluator.evaluate(new_predictions)
    
    # Check if performance improved
    if new_auc > auc:
        # Log new model
        with mlflow.start_run():
            mlflow.log_metric("auc", new_auc)
            mlflow.spark.log_model(new_rf_model, "random_forest_model")
        
        # Register and transition model
        # ...

# Schedule the job using Databricks Jobs
```

**Expected Result**: The model retraining and deployment are automated, ensuring the model stays up-to-date.

---

## Step 10: Clean Up Resources

**Objective**: Release resources and clean up the environment.

- **Stop or Terminate Clusters**:
  - In the **Clusters** tab, click **Terminate** on your cluster if it's no longer needed.

- **Disable Model Serving**:
  - In the **Models** tab, disable serving for your model if no longer needed.

- **Delete Unused Data and Models**:
  - Remove any data and models stored during the exercise if they're no longer needed.

**Expected Result**: Resources are cleaned up, preventing unnecessary costs.

---

## Summary of the Activity

In this comprehensive hands-on session, you have learned how to:

1. **Set up a Databricks workspace** with MLflow for MLOps tasks.
2. **Train a machine learning model** and prepare it for deployment.
3. **Track experiments** using MLflow, logging parameters, metrics, and models.
4. **Register and manage models** using the MLflow Model Registry.
5. **Deploy the model** using MLflow Model Serving and test it via REST API.
6. **Set up model monitoring**, logging inference metrics over time.
7. **Configure alerts** to notify when model performance degrades.
8. **Automate retraining and deployment**, ensuring the model remains effective.
9. **Clean up resources** to maintain an efficient environment.

This exercise demonstrates how Databricks and MLflow can be leveraged for MLOps practices, providing practical experience in deploying and monitoring machine learning models in a production environment.

---

## Additional Resources

- **Databricks Documentation**:
  - [MLflow Tracking](https://docs.databricks.com/applications/mlflow/index.html)
  - [Model Serving](https://docs.databricks.com/applications/mlflow/model-serving.html)
- **MLflow Documentation**:
  - [MLflow Models](https://www.mlflow.org/docs/latest/models.html)
  - [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- **Books**:
  - *"Introducing MLOps: How to Scale Machine Learning in the Enterprise"* by Mark Treveil et al.
- **Articles and Tutorials**:
  - [Operationalizing Machine Learning with MLOps on Databricks](https://databricks.com/blog/2020/12/16/operationalizing-machine-learning-with-mlops-on-databricks.html)
  - [Monitoring Machine Learning Models in Production](https://towardsdatascience.com/monitoring-machine-learning-models-in-production-634f27df0c09)
- **Videos**:
  - [MLOps with MLflow and Databricks](https://www.youtube.com/watch?v=_9Vqp0UO2lU)

Feel free to explore further and apply these techniques to other datasets and real-world problems. Embracing MLOps practices will enhance the reliability and scalability of your machine learning projects.

---

**Note**: Ensure that you have the necessary permissions and resources available in your Databricks environment to execute the deployment and monitoring steps, as they may require additional configurations or subscriptions. If you encounter any issues, refer to the Databricks documentation or consult your system administrator.

---
# Hands-on with Databricks and Delta Lake

This exercise will guide you through deploying and managing an **Apache Spark cluster** using **Databricks Community Edition**, 
as well as ingesting and processing data with **Delta Lake**.<br />
You will explore setting up a workspace, loading data, running a Spark job, and managing data processing using Delta Lake.

## Objective

Set up a **Databricks Community Edition workspace**, manage it to process a sample dataset using **Delta Lake**, and explore real-time data transformations, 
optimization, and management techniques.

### Step-by-step guide

---

### **Step 1: Set up Databricks Community Edition**

**Objective**: Create a Databricks account and launch a cluster to process data.

- **Create an account**:
  - Go to the [Databricks Community Edition](https://community.cloud.databricks.com/) and sign up with your email.
  - Once you complete the signup, you’ll be directed to your Databricks workspace.

- **Launch a Cluster**:
  - In the workspace, navigate to the **Clusters** tab.
  - Click on **Create Cluster** and configure the cluster:
    - **Cluster Name**: Choose any name for your cluster (e.g., `my-cluster`).
    - **Cluster Mode**: Choose **Standard**.
    - **Databricks Runtime Version**: Choose a runtime with Delta Lake (e.g., **7.3 LTS (Scala 2.12, Spark 3.0.1)**).
    - **Driver Type**: Select a standard driver type (Community Edition has limited options).
  - Click **Create Cluster** to start your cluster (it may take a few minutes to initialize).

**Expected result**: A fully provisioned Databricks cluster ready for data ingestion and processing.

---

### **Step 2: Upload Data to DBFS (Databricks File System)**

**Objective**: Upload a sample dataset (e.g., a CSV file) to the Databricks file system (**DBFS**).

- **Example Dataset**: Download a dataset from [Kaggle](https://www.kaggle.com/datasets) or use any CSV file you have available.

- **Upload Data**:
  - In the Databricks UI, navigate to the **Data** tab.
  - Click **Add Data** > **Upload File** and select your CSV file.
  - Once uploaded, Databricks will store the data in **DBFS**.
  - You can access the file at a path like `/FileStore/tables/your_file.csv`.

**Expected result**: Your dataset is now uploaded to **DBFS** and ready for data processing within your Databricks environment.

---

### **Step 3: Process the Data using Delta Lake**

**Objective**: Process and store the data in **Delta Lake** by running a Spark job using **PySpark**.

- **Create a New Notebook**:
  - In the Databricks workspace, navigate to the **Workspace** tab and click **Create** > **Notebook**.
  - Select **Python** as the language and name your notebook.

- **Run a PySpark Job**:
  - Use **PySpark** to load the data, process it, and store the results as a Delta table.

  ```python
  from pyspark.sql import SparkSession
  from pyspark.sql.functions import col
  
  # Create Spark session
  spark = SparkSession.builder.appName("Data Processing").getOrCreate()
  
  # Load data from DBFS
  df = spark.read.csv("/FileStore/tables/your_file.csv", header=True, inferSchema=True)
  
  # Perform basic transformation
  df_filtered = df.filter(col("column_name") > 100)  # Example filtering
  
  # Write to Delta Lake
  df_filtered.write.format("delta").save("/tmp/delta/my_filtered_data")

- **Monitor Job Execution**:
  - Monitor the execution of the job in the Notebook UI. Check the job logs and progress in real-time.

**Expected result**: The Spark job processes the data and writes the filtered data to a Delta Lake table.

---

### **Step 4: Query the Delta Lake Table and Explore Time Travel**

**Objective**: Use Delta Lake’s time travel feature to query previous versions of the data and explore schema enforcement.

- **Query the Delta Table**:

After creating the Delta table in the previous step, you can query the table to retrieve specific data versions using Delta’s time travel feature.
```python
# Read Delta table
df_delta = spark.read.format("delta").load("/tmp/delta/my_filtered_data")

# Use time travel to query the previous version of the table
df_previous = spark.read.format("delta").option("versionAsOf", 0).load("/tmp/delta/my_filtered_data")
df_previous.show()
```

- **Schema Enforcement**:

Try writing data that violates the schema (e.g., inserting a row with missing fields) and observe how Delta Lake prevents the schema violation.

**Expected result**: Successfully query historical data and demonstrate Delta Lake’s time travel feature, along with schema enforcement that maintains data quality.

---

### **Step 5: Optimize and Manage Data in Delta Lake**

**Objective**: Optimize the Delta Lake table for efficient storage and querying.

- **Optimize the Table**:

Use Delta Lake’s Optimize command to compact small files and improve query performance.
```python
spark.sql("OPTIMIZE '/tmp/delta/my_filtered_data'")
```

- **Query Partitioned Data**:

If your dataset is large, use partitioning to enhance query speed by limiting the data read.
```python
df.write.format("delta").partitionBy("column_name").save("/tmp/delta/my_partitioned_data")
```

**Expected result**: The data is optimized for both storage and query performance through compaction and partitioning.

---

**Step 6: Manage and Monitor the Databricks Cluster**

**Objective**: Manage your cluster for efficient performance and cost management.

- **Auto-scaling**:
Databricks Community Edition doesn’t support auto-scaling, but you can manually stop and start clusters based on your workload needs.
Monitor Cluster Health:
- Use the Clusters tab to monitor cluster performance, job execution, and resource utilization.
- View detailed logs and metrics for the running cluster in the Spark UI. 

**Expected result**: The cluster is efficiently managed and monitored for performance and resource utilization.

---

### **Summary of the activity**

In this hands-on session, you have learned how to:

1. Set up a Databricks Community Edition workspace and launch an Apache Spark cluster.
2. Upload and manage data in DBFS and process it using Delta Lake.
3. Run Spark jobs with PySpark to transform and query data.
4. Use Delta Lake to optimize storage, query performance, and enable time travel.
5. Manage and monitor the Databricks cluster for performance and efficiency.

This hands-on session demonstrates the power of Databricks and Delta Lake for handling large-scale data processing, optimizing 
query performance, and managing cloud-based data pipelines efficiently.


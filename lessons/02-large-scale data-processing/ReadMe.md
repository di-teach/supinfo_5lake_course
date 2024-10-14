# Hands-on: Large-Scale data processing and performance optimization with Databricks and Delta Lake

This hands-on exercise will guide you through ingesting, processing, and optimizing large datasets using **Databricks**, 
**Apache Spark**, and **Delta Lake**. You'll set up a Databricks workspace, create scalable clusters, ingest big data, 
perform distributed data processing, and apply performance optimization techniques such as partitioning, caching, and query optimization.

## Objective

Set up a **Databricks** workspace, create and manage clusters for large-scale data processing, utilize **Apache Spark** to 
process and analyze large datasets, and implement performance optimization techniques using **Delta Lake** features.

### Step-by-Step Guide

---

### **Step 1: Set Up Your Databricks Workspace**

Check that your Databricks workspace is set up (if you have not done so already during a previous exercise).

---

### **Step 2: Create a Scalable Cluster**

**Objective**: Configure and launch a cluster capable of handling large datasets and optimization tasks.

- **Create a New Cluster**:
  - Navigate to the **Clusters** tab and click **Create Cluster**.
  - Configure your cluster:
    - **Cluster Name**: Give your cluster a descriptive name (e.g., `large-scale-optimization-cluster`).
    - **Cluster Mode**: Choose **Standard**.
    - **Databricks Runtime Version**: Select the latest LTS version with Delta Lake support (e.g., **10.4 LTS**).
    - **Autoscaling**: Enable autoscaling to allow the cluster to adjust based on workload.
      - Set the **Min Workers** and **Max Workers** (e.g., Min: 2, Max: 8).
    - **Worker Type**: Select an instance type appropriate for your workload (e.g., **Standard_DS3_v2** on Azure).
    - **Driver Type**: Typically matches the worker type.
  - **Advanced Options** (optional):
    - Configure **Spark Configurations** if necessary.
  - Click **Create Cluster**.

**Expected Result**: A scalable cluster is created and ready to process large datasets and perform optimization tasks.

---

### **Step 3: Ingest a Large Dataset**

**Objective**: Import a large dataset into Databricks for processing and optimization.

- **Select a Dataset**:
  - Use a publicly available large dataset, such as the **New York City Taxi Trip Data** or a large dataset from [Kaggle](https://www.kaggle.com/datasets).
  - Ensure the dataset is sizable to simulate large-scale processing (e.g., several GBs).

- **Ingest Data into Databricks**:
  - **Option 1: Upload to DBFS** (for smaller large datasets):
    - In the **Data** tab, click **Add Data** > **Upload File**.
    - Upload the dataset file.
  - **Option 2: External Data Sources**:
    - For very large datasets, read directly from cloud storage (e.g., AWS S3, Azure Blob Storage).
    - Ensure your cluster has access permissions to the storage.

- **Access the Data in a Notebook**:
  - Create a new notebook or use an existing one.
  - Use **Spark** to read the data:
    ```python
    # For CSV data
    df = spark.read.format("csv").option("header", "true").load("path_to_your_data")
    ```
    Replace `"path_to_your_data"` with the appropriate file path or storage URI.

**Expected Result**: The large dataset is successfully ingested and accessible within your Databricks notebook.

---

### **Step 4: Process the Data Using Apache Spark**

**Objective**: Perform distributed data processing on the large dataset using Apache Spark.

- **Data Exploration**:
  - Examine the schema and preview the data:
    ```python
    df.printSchema()
    df.show(5)
    ```
  - Count the number of records:
    ```python
    record_count = df.count()
    print(f"Total records: {record_count}")
    ```

- **Data Transformation**:
  - Perform transformations such as filtering, aggregations, and column manipulations.
    ```python
    # Example: Filter out records with null values in a specific column
    df_clean = df.filter(df["your_column"].isNotNull())
    ```
  - Aggregate data to extract insights.
    ```python
    # Example: Calculate average value grouped by a category
    df_grouped = df_clean.groupBy("category_column").avg("numeric_column")
    df_grouped.show()
    ```

- **Optimizing Transformations**:
  - Repartition data for better parallelism.
    ```python
    df_repartitioned = df_clean.repartition(8)  # Adjust based on your cluster size
    ```

**Expected Result**: Data transformations are applied efficiently across the cluster using Spark's distributed processing capabilities.

---

### **Step 5: Write the Processed Data to Delta Lake**

**Objective**: Save the transformed data in Delta Lake format to enable advanced features like ACID transactions and time travel.

- **Write Data to Delta Lake**:
  - Save the processed data in Delta format:
    ```python
    df_grouped.write.format("delta").mode("overwrite").save("/mnt/delta/processed_data")
    ```

- **Register as a Table (Optional)**:
  - Create a table in the Metastore for easy access via SQL:
    ```python
    spark.sql("CREATE TABLE processed_data USING DELTA LOCATION '/mnt/delta/processed_data'")
    ```

**Expected Result**: The processed data is saved in Delta Lake format and accessible for further analysis.

---

### **Step 6: Analyze the Processed Data**

**Objective**: Perform analysis on the processed data to extract meaningful insights.

- **Use Spark SQL for Analysis**:
  - Run SQL queries on the processed data:
    ```python
    df_analysis = spark.sql("""
    SELECT category_column, AVG(numeric_column) as average_value
    FROM processed_data
    GROUP BY category_column
    ORDER BY average_value DESC
    """)
    df_analysis.show()
    ```

- **Visualize Results**:
  - Use Databricks' built-in visualization tools:
    - After running the query, click on the chart icon above the result set.
    - Choose the appropriate chart type (e.g., bar chart) and configure settings.

**Expected Result**: Insights are derived from the processed data, and results are visualized.

---

### **Step 7: Implement Performance Optimization Techniques**

**Objective**: Optimize data processing and querying performance using Delta Lake features.

#### **a. Partitioning Data**

- **Re-Partition the Data**:
  - Write data partitioned by a relevant column to improve query performance:
    ```python
    df_grouped.write.format("delta").mode("overwrite").partitionBy("category_column").save("/mnt/delta/processed_data_partitioned")
    ```

- **Update the Table Reference**:
  - Drop the existing table and create a new one pointing to the partitioned data:
    ```python
    spark.sql("DROP TABLE IF EXISTS processed_data_partitioned")
    spark.sql("CREATE TABLE processed_data_partitioned USING DELTA LOCATION '/mnt/delta/processed_data_partitioned'")
    ```

- **Run Queries and Compare Performance**:
  - Execute queries filtering on the partitioned column and measure execution times.
    ```python
    import time
    start_time = time.time()
    result = spark.sql("SELECT * FROM processed_data_partitioned WHERE category_column = 'YourValue'").collect()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    ```

**Expected Result**: Improved query performance when filtering on the partitioned column due to partition pruning.

#### **b. Optimize Data with Z-Ordering**

- **Apply Z-Ordering**:
  - Optimize the data layout to improve performance for queries filtering on multiple columns:
    ```python
    spark.sql("OPTIMIZE processed_data_partitioned ZORDER BY (other_column)")
    ```

- **Run Complex Queries**:
  - Execute queries filtering on multiple columns and measure execution times.
    ```python
    start_time = time.time()
    result = spark.sql("""
    SELECT * FROM processed_data_partitioned 
    WHERE category_column = 'YourValue' AND other_column BETWEEN value1 AND value2
    """).collect()
    end_time = time.time()
    print(f"Z-Ordered Query Time taken: {end_time - start_time} seconds")
    ```

**Expected Result**: Faster query execution when filtering on multiple columns, thanks to Z-Ordering.

#### **c. Compact Small Files**

- **Use Optimize Command**:
  - Compact small files to reduce overhead:
    ```python
    spark.sql("OPTIMIZE processed_data_partitioned")
    ```

- **Compare Execution Times**:
  - Observe performance improvements due to reduced file overhead.

**Expected Result**: Improved query performance and reduced storage overhead.

#### **d. Implement Caching**

- **Cache Frequently Accessed Data**:
  - Cache the table or specific DataFrames to speed up repetitive queries:
    ```python
    spark.sql("CACHE TABLE processed_data_partitioned")
    ```

- **Run Repetitive Queries**:
  - Execute the same query multiple times and observe execution times.
    ```python
    for i in range(3):
        start_time = time.time()
        result = spark.sql("""
        SELECT * FROM processed_data_partitioned 
        WHERE category_column = 'YourValue' AND other_column BETWEEN value1 AND value2
        """).collect()
        end_time = time.time()
        print(f"Iteration {i+1}: Time taken: {end_time - start_time} seconds")
    ```

**Expected Result**: Significantly faster query execution on subsequent runs due to data being cached.

#### **e. Tune Spark Configurations**

- **Adjust Shuffle Partitions**:
  - Optimize the number of shuffle partitions based on data size and cluster resources:
    ```python
    spark.conf.set("spark.sql.shuffle.partitions", "200")  # Adjust as needed
    ```

- **Run Queries and Observe Performance**:
  - Measure execution times to see the impact of configuration changes.

**Expected Result**: Enhanced performance due to better resource utilization.

---

### **Step 8: Monitor and Adjust Cluster Resources**

**Objective**: Monitor cluster performance and adjust resources as needed.

- **Monitor Cluster Metrics**:
  - In the **Clusters** tab, view the cluster's **Metrics** to monitor CPU, memory, and disk usage.
  - Use the **Spark UI** to analyze job and stage execution details.

- **Adjust Cluster Size**:
  - Scale the cluster up or down based on performance observations and workload demands.

**Expected Result**: The cluster is optimized for performance and cost-effectiveness.

---

### **Step 9: Clean Up Resources**

**Objective**: Release resources and clean up the environment to avoid unnecessary costs.

- **Uncache Tables**:
  ```python
  spark.sql("UNCACHE TABLE processed_data_partitioned")
  ```

- **Terminate Clusters**:
  - In the **Clusters** tab, click **Terminate** on your cluster when it's no longer needed.

- **Delete Unused Data**:
  - Remove any data stored during the exercise if it's no longer needed:
    ```python
    dbutils.fs.rm("/mnt/delta/processed_data", recurse=True)
    dbutils.fs.rm("/mnt/delta/processed_data_partitioned", recurse=True)
    ```

**Expected Result**: Resources are cleaned up, preventing unnecessary costs.

---

### **Summary of the Activity**

In this comprehensive hands-on session, you have learned how to:

1. Set up a Databricks workspace and create clusters for large-scale data processing and optimization.
2. Ingest large datasets into Databricks from various sources.
3. Utilize Apache Spark for distributed data processing and transformations.
4. Save processed data in Delta Lake format to leverage advanced features.
5. Analyze and visualize processed data to extract insights.
6. Implement performance optimization techniques:
  - **Partitioning data** to improve query performance.
  - **Using Z-Ordering** to optimize data layout.
  - **Compacting small files** to reduce overhead.
  - **Caching data** to speed up repetitive queries.
  - **Tuning Spark configurations** for optimal resource utilization.
7. Monitor cluster performance and adjust resources accordingly.
8. Clean up resources to manage costs effectively.

This exercise demonstrates how Databricks, Apache Spark, and Delta Lake can efficiently handle large-scale data processing tasks and how performance optimization techniques can significantly enhance efficiency in big data environments.

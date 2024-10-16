# Datalake and Databricks for data science module setup

This guide will help you set up the environment needed for the **Cloud Data Services** module.<br />  
We will ensure that you have access to **Databricks Community Edition**, **Python**, and other necessary tools to complete the course successfully.

---

## Prerequisites

For this course, you will need the following tools and environments:
- **Databricks Community Edition account** for cloud data services.
- **Python** for scripting and data processing.
- **Docker** to containerize and run services locally.
- **Git** and **Bash** for command-line operations.

---

## Step 1: Installing Bash and Git

### Bash
Check that you have bash installed in your machine by opening a terminal and running:

```bash
bash --version
```
This should be the case for all Mac and Linux users.

If you are using Windows, you can use the Git Bash terminal that comes with Git. You can download it [here](https://git-scm.com/downloads).

---

## Step 2: Installing Python

You will use **Python** for data processing and scripting within this module. Follow these steps to install Python:

1. **Check if Python is installed**:
Run the following command to check if Python is installed:
```bash
python --version
``` 

2. **Installing Python**:
If Python is not installed, follow the official installation instructions for your operating system [here](https://www.python.org/downloads/).

---

## Step 3: Installing Docker

Check that you have Docker desktop installed in your machine by running:

```bash
docker -v
```

If that is not the case, follow the official installation instructions for your operating system [here](https://docs.docker.com/desktop/).
For those of you working on Windows, you might need to update Windows Subsystem for Linux. To do so, simply open PowerShell and run:

```bash
wsl --update
```

Once docker is installed, make sure that it is running correctly by running:

```bash
docker run -p 80:80 docker/getting-started
```

If you check the Docker App, you should see a getting started container running. Once you've checked that this works correctly, remove the container via the UI.

<details>
    <summary><b>Optional</b></summary>
    You can also perform these operations directly from the command line, by running <code>docker ps</code> to check the running containers and <code>docker rm -f [CONTAINER-ID]</code> to remove it.
</details>

---

### Step 4: Creating a Databricks Community Edition Account

For this course, we will use **Databricks Community Edition** for data processing and cluster management. Follow these steps to create an account:

#### Sign up for Databricks Community Edition:

- Visit [Databricks Community Edition](https://community.cloud.databricks.com/signup) and sign up for a free account.
- Databricks Community Edition provides a free, hosted environment for running Apache Spark jobs and Delta Lake features.

#### Login and create a cluster:

- After logging into **Databricks Community Edition**, navigate to the **Clusters** tab.
- Create a new cluster, select the latest **Databricks Runtime Version**, and configure basic settings (e.g., name your cluster, choose instance types).
- Start the cluster and ensure it is running correctly before proceeding with your tasks.

#### Upload data:

- You can upload sample datasets (e.g., CSV files) to **DBFS** (Databricks File System) via the **Data** tab in the Databricks UI.
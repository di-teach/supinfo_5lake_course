# Hands-On: Implementing CI/CD for Data Science Projects with Databricks

This hands-on exercise will guide you through setting up a Continuous Integration and Continuous Deployment (CI/CD) pipeline for a data science project using **Databricks**, **GitHub Actions**, and **MLflow**. You will learn how to automate testing, versioning, and deployment of machine learning models, streamlining the delivery process and ensuring code and model quality.

## Objective

- Integrate **Databricks Repos** with **GitHub** for version control.
- Write **unit tests** for your data science code.
- Use **MLflow Model Registry** for model versioning.
- Set up a **CI/CD pipeline** using **GitHub Actions**.
- Automate deployment of notebooks and models to Databricks.
- Validate the deployment and ensure the CI/CD pipeline works effectively.

---

## Step 1: Set Up Version Control with Databricks Repos and GitHub

**Objective**: Integrate your Databricks workspace with GitHub to manage code versioning.

### a. Create a GitHub Repository

- **Create a new repository** on GitHub (e.g., `databricks-ci-cd-example`).
- **Clone the repository** to your local machine.

### b. Structure Your Project

Create the following directory structure:

```
databricks-ci-cd-example/
├── notebooks/
│   └── train_model.py
├── tests/
│   └── test_train_model.py
├── .github/
│   └── workflows/
│       └── ci_cd_pipeline.yml
├── requirements.txt
└── README.md
```

### c. Add Initial Code

In `notebooks/train_model.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and model with MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

if __name__ == "__main__":
    main()
```

In `requirements.txt`:

```
scikit-learn
mlflow
pytest
```

### d. Push Code to GitHub

- **Commit and push** your code to the GitHub repository.

### e. Set Up Databricks Repos

- **In Databricks**, go to **Repos**.
- **Click "Add Repo"** and clone your GitHub repository using HTTPS or SSH.
- **Authenticate** if necessary.

**Expected Result**: Your code is version-controlled and accessible within Databricks Repos.

---

## Step 2: Write and Run Unit Tests

**Objective**: Implement unit tests to ensure your code works as expected.

### a. Write Tests Using Pytest

In `tests/test_train_model.py`:

```python
from notebooks.train_model import main
import os

def test_model_training():
    # Run the main function
    main()
    # Check if MLflow run exists
    assert os.path.exists("mlruns"), "MLflow run not found"

    # Additional tests can be added here
```

### b. Run Tests Locally

- **Install dependencies**:

  ```bash
  pip install -r requirements.txt
  ```

- **Run tests**:

  ```bash
  pytest tests/
  ```

**Expected Result**: Tests pass, confirming that your code works as expected.

---

## Step 3: Use MLflow Model Registry for Model Versioning

**Objective**: Register your model with MLflow Model Registry in Databricks.

### a. Modify Training Script to Register Model

Update `notebooks/train_model.py`:

```python
import mlflow
import mlflow.sklearn

def main():
    # ... (previous code)

    # Log metrics and model with MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Register the model
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/random_forest_model"
    mlflow.register_model(model_uri, "IrisRFModel")

if __name__ == "__main__":
    main()
```

### b. Run Training Script in Databricks

- **In Databricks**, open `notebooks/train_model.py`.
- **Run the script** in a notebook cell.
- **Check MLflow**:

  - Go to **Experiments** in Databricks.
  - Find your run and verify that the model is registered.

**Expected Result**: The model is registered in MLflow Model Registry with versioning.

---

## Step 4: Set Up a CI/CD Pipeline with GitHub Actions

**Objective**: Automate testing and deployment using GitHub Actions.

### a. Create GitHub Actions Workflow

In `.github/workflows/ci_cd_pipeline.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install databricks-cli

      - name: Run tests
        run: pytest tests/

      - name: Configure Databricks CLI
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          databricks configure --token <<EOF
          $DATABRICKS_HOST
          $DATABRICKS_TOKEN
          EOF

      - name: Upload Notebooks to Databricks
        run: |
          databricks workspace import_dir notebooks /Repos/your_user/databricks-ci-cd-example/notebooks -o

      - name: Trigger Databricks Job
        run: |
          databricks jobs run-now --job-id <job-id>
```

**Note**:

- Replace `<job-id>` with your Databricks job ID.
- Set **GitHub Secrets** for `DATABRICKS_HOST` and `DATABRICKS_TOKEN`.

### b. Set Up a Job in Databricks

- **Create a job** in Databricks:

  - Go to **Jobs** > **Create Job**.
  - **Configure** the job to run `train_model.py`.
  - **Note the Job ID**.

### c. Push Changes and Trigger the Pipeline

- **Commit and push** your changes to GitHub.
- **Check GitHub Actions**:

  - Go to the **Actions** tab in your repository.
  - **Monitor** the pipeline execution.

**Expected Result**: The CI/CD pipeline runs tests, uploads code to Databricks, and triggers the job automatically.

---

## Step 5: Validate Deployment

**Objective**: Ensure that the automated deployment works correctly.

### a. Verify Model Registration

- **Check MLflow Model Registry** in Databricks.
- **Confirm** that a new model version is registered.

### b. Test Model Serving (Optional)

- **Set up Model Serving** for your registered model.
- **Send test requests** to the model endpoint.

**Expected Result**: The model is deployed, and you can interact with it via API.

---

## Step 6: Clean Up Resources

**Objective**: Release resources and clean up the environment.

- **Stop or terminate clusters** if not needed.
- **Delete test models** from MLflow if desired.
- **Disable Model Serving** if enabled.

---

## Summary of the Activity

In this hands-on exercise, you have:

1. **Integrated Databricks Repos** with GitHub for version control.
2. **Written and run unit tests** for your data science code.
3. **Used MLflow Model Registry** to version and manage models.
4. **Set up a CI/CD pipeline** using GitHub Actions.
5. **Automated the deployment** of notebooks and models to Databricks.
6. **Validated the deployment**, ensuring the CI/CD pipeline works effectively.

---

## Additional Resources

- **Databricks Documentation**:

  - [Databricks Repos](https://docs.databricks.com/repos.html)
  - [MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html)
  - [Databricks CLI](https://docs.databricks.com/dev-tools/cli/index.html)

- **GitHub Actions**:

  - [GitHub Actions Documentation](https://docs.github.com/en/actions)
  - [Using GitHub Actions for Python CI/CD](https://docs.github.com/en/actions/guides/building-and-testing-python)

- **Articles and Tutorials**:

  - [CI/CD for Machine Learning using Databricks and Azure DevOps](https://databricks.com/blog/2019/08/14/continuous-integration-continuous-delivery-of-etl-pipelines-with-databricks-and-azure-devops.html)
  - [Automating Machine Learning Workflows with GitHub Actions](https://towardsdatascience.com/automating-machine-learning-workflows-with-github-actions-8c693e7e520f)

---

**Note**: Ensure that you have the necessary permissions and resources available in your Databricks environment to execute the steps, and that your GitHub repository is properly configured for GitHub Actions.

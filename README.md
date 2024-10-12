# Cloud Data Services module setup

This guide will help you set up the environment needed for the **Cloud Data Services** module.<br />
We will ensure that you have access to **Azure**, **Python**, and other necessary tools to complete the course successfully.

---

## Prerequisites

For this course, you will need the following tools and environments:
- **Microsoft Azure account** for deploying cloud data services.
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

## Step 4: Creating a Microsoft Azure Account

If you don't have an Azure account, follow these steps to create one. Azure provides a **free tier** which includes limited usage of services like **Azure Data Factory**, **Synapse Analytics**, and **Azure Blob Storage**.

### Azure Account Creation

1. **Sign up for Azure**:
    - Visit [Azure Free Account Sign Up](https://azure.microsoft.com/en-us/free/) and sign up for a free account.
    - You will need to provide a valid credit card for identity verification. No charges will be applied to your account unless you exceed the free tier limits.

2. **Activate free services**:
    - Once signed up, you can activate your free services, which include **750 hours of virtual machines**, **5 GB of Blob storage**, and **free use of Data Factory** and **Synapse Analytics** for a limited time.

3. **Install the Azure CLI** (optional):
    - To manage Azure services from the command line, you can install the **Azure CLI**. Instructions can be found [here](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).

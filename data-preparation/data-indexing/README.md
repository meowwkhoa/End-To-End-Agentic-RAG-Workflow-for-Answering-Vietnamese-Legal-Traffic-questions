# Data Ingestion Pipeline Setup Guide

This guide provides step-by-step instructions to set up the Data Ingestion Pipeline environment, install dependencies, and launch Jupyter Lab. Follow the instructions below to get started.

---

## 1. Create and Activate the Conda Environment

First, create a new Conda environment with Python 3.9:

```bash
conda create -n Data_Ingestion_Pipeline python=3.9
```

Activate the newly created environment:
```bash
conda activate Data_Ingestion_Pipeline
```
## 2. Install Required Dependencies
Install all necessary packages using pip:
```bash
pip install -r requirements.txt
```

## 3. Launch Jupyter Lab
Start Jupyter Lab to begin working with your notebooks:
```bash
jupyter lab
```
## 4. Kubernetes Configuration
To work with the Weaviate service within your Kubernetes cluster, follow these steps:

1. Set the Namespace:

- Switch to the `weaviate` namespace:
    ```bash
    kubens weaviate
    ```
2. Port-Forward the Weaviate Service:

- Forward port `85` of the Weaviate service to local port `8085`:
    ```bash
    kubectl port-forward svc/weaviate 8085:85
    ```

## 5. Data indexing
- Run the the cells of the `notebook.ipynb`.
# Project Setup Guide

## Table of Contents
- [Service Account Setup](#service-account-setup)
- [IAM Configuration](#iam-configuration)
- [Environment Setup](#environment-setup)
- [GCE Instance Creation](#gce-instance-creation)
- [SSH Configuration](#ssh-configuration)
- [Jenkins Deployment](#jenkins-deployment)
- [Kubernetes Configuration](#kubernetes-configuration)

---

## Service Account Setup
Create a Service Account and generate a key:
- Add a file key in JSON format:
     - **Path:** `service account -> create service account -> add key`

---

## IAM Configuration
Grant Compute Admin role to the service account:
1. Navigate to **IAM & Admin > IAM**
2. Add the service account to IAM
3. Assign role: **Compute Admin**

---

## Environment Setup
1. Create Conda environment:
    ```bash
      conda create -n ansible python==3.9 -y
      conda activate ansible
    ```

2. Install dependencies:
    ```bash
      pip install -r requirements.txt
    ```

---

## GCE Instance Creation
1. Authenticate and set project:
    ```bash
      gcloud auth application-default login
      gcloud config set project jenkins1-433523
    ```

2. Enable Compute Engine API

3. Create instance via Ansible:
    ```bash
      ansible-playbook playbooks/create_compute_instance.yaml
    ```

---

## SSH Configuration
1. Update SSH keys in inventory:
    ```bash
      cat ~/.ssh/id_rsa.pub
    ```
2. Add the SSH public key to:
  GCP Console > Compute Engine > Metadata > SSH Keys

---
## Jenkins deployment
Run the Jenkins deployment playbook:
  ```bash
      ansible-playbook -i inventory playbooks/deploy_jenkins.yaml
  ```

--- 
## Kubernetes Configuration
1. Export Kubernetes certificate:
    ```bash
      kubectl config view --raw > ~/.kube/config
    ```

2. Create cluster role bindings:
    ```bash
      kubectl create clusterrolebinding rag-controller-admin-binding \
      --clusterrole=admin \
      --serviceaccount=rag-controller:default \
      --namespace=rag-controller

      kubectl create clusterrolebinding anonymous-admin-binding \
        --clusterrole=admin \
        --user=system:anonymous \
        --namespace=rag-controller
    ```

---
## Important notes
- Replace `jenkins1-433523` with your actual GCP project ID
- Ensure firewall rules allow necessary traffic (SSH/HTTP/HTTPS)





   

# GCP Deployment Guide for Student Dropout Prediction Model

This document provides a step-by-step guide for deploying the Student Dropout Prediction model to Google Cloud Platform (GCP) using Google Kubernetes Engine (GKE).

## Prerequisites

Before starting the deployment process, ensure you have the following:

1. **Google Cloud Platform Account**: You need a GCP account with billing enabled.
2. **Google Cloud SDK**: Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) on your local machine.
3. **kubectl**: Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for interacting with Kubernetes clusters.
4. **Docker**: Install [Docker](https://docs.docker.com/get-docker/) for building and pushing container images.
5. **Trained Model**: Ensure you have a trained model saved in the `models/` directory.

## Step 1: Set Up GCP Project

1. Create a new GCP project or use an existing one:

```bash
# Create a new project
gcloud projects create student-dropout-predictor --name="Student Dropout Predictor"

# Set the project as active
gcloud config set project student-dropout-predictor
```

2. Enable the required APIs:

```bash
# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# Enable Kubernetes Engine API
gcloud services enable container.googleapis.com
```

## Step 2: Create a GKE Cluster

Create a Kubernetes cluster in GCP:

```bash
# Create a cluster with 2 nodes
gcloud container clusters create student-dropout-cluster \
    --zone us-central1-a \
    --num-nodes 2 \
    --machine-type e2-standard-2
```

Get credentials for kubectl:

```bash
gcloud container clusters get-credentials student-dropout-cluster --zone us-central1-a
```

## Step 3: Build and Push Docker Image

1. Build the Docker image:

```bash
# Navigate to the project root directory
cd /path/to/capstone-2

# Build the Docker image
docker build -t gcr.io/student-dropout-predictor/student-dropout-predictor:v1 .
```

2. Push the image to Google Container Registry:

```bash
# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker

# Push the image
docker push gcr.io/student-dropout-predictor/student-dropout-predictor:v1
```

## Step 4: Update Kubernetes Manifests

Update the `deployment.yaml` file to use your GCP project ID:

```bash
# Replace PROJECT_ID with your actual project ID
sed -i "s/PROJECT_ID/student-dropout-predictor/g" k8s/deployment.yaml
```

## Step 5: Deploy to Kubernetes

Apply the Kubernetes manifests:

```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Apply service
kubectl apply -f k8s/service.yaml

# Apply HPA
kubectl apply -f k8s/hpa.yaml
```

Check the deployment status:

```bash
# Check deployment status
kubectl get deployments

# Check pods
kubectl get pods

# Check service
kubectl get services
```

## Step 6: Access the Prediction Service

Once the service is deployed, you can access it using the external IP assigned to the LoadBalancer service:

```bash
# Get the external IP
kubectl get service student-dropout-predictor
```

The service will be available at `http://<EXTERNAL_IP>`.

## Step 7: Test the Prediction Service

Use the `test_prediction.py` script to test the deployed service:

```bash
python scripts/test_prediction.py --url http://<EXTERNAL_IP> --test all
```

## Step 8: Set Up Monitoring and Logging

1. Enable Cloud Monitoring and Logging:

```bash
# Enable Monitoring API
gcloud services enable monitoring.googleapis.com

# Enable Logging API
gcloud services enable logging.googleapis.com
```

2. View logs in the GCP Console:
   - Go to the GCP Console
   - Navigate to Kubernetes Engine > Workloads
   - Select the `student-dropout-predictor` deployment
   - Click on "Logs" to view the logs

3. Set up a monitoring dashboard:
   - Go to the GCP Console
   - Navigate to Monitoring > Dashboards
   - Create a new dashboard
   - Add charts for CPU usage, memory usage, and request latency

## Step 9: Set Up Autoscaling

The Horizontal Pod Autoscaler (HPA) is already configured in `hpa.yaml`. You can check its status:

```bash
kubectl get hpa
```

To test autoscaling, you can run a load test:

```bash
python scripts/test_prediction.py --url http://<EXTERNAL_IP> --test load --num-requests 1000 --concurrency 20
```

## Step 10: Clean Up Resources

When you're done with the deployment, you can clean up the resources to avoid incurring charges:

```bash
# Delete the Kubernetes resources
kubectl delete -f k8s/

# Delete the GKE cluster
gcloud container clusters delete student-dropout-cluster --zone us-central1-a

# Delete the container images
gcloud container images delete gcr.io/student-dropout-predictor/student-dropout-predictor:v1 --force-delete-tags
```

## Troubleshooting

### Common Issues

1. **Image Pull Errors**:
   - Ensure you've configured Docker to use gcloud as a credential helper
   - Check that the image exists in the Container Registry

2. **Pod Startup Failures**:
   - Check the pod logs: `kubectl logs <pod-name>`
   - Ensure the model file exists in the container

3. **Service Not Accessible**:
   - Check if the service has an external IP: `kubectl get service student-dropout-predictor`
   - Ensure the pods are running: `kubectl get pods`

4. **Autoscaling Not Working**:
   - Check the HPA status: `kubectl get hpa`
   - Ensure the metrics server is running: `kubectl get deployment metrics-server -n kube-system`

## Conclusion

You have successfully deployed the Student Dropout Prediction model to GCP using Google Kubernetes Engine. The model is now accessible via a RESTful API and can be used to make predictions in real-time.

For more information on managing and scaling your deployment, refer to the [GKE documentation](https://cloud.google.com/kubernetes-engine/docs).
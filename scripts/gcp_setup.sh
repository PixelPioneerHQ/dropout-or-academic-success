#!/bin/bash
# Script to set up GCP project and deploy to GKE

# Exit on error
set -e

# Configuration
PROJECT_ID="student-dropout-predictor"  # Replace with your GCP project ID
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="student-dropout-cluster"
IMAGE_NAME="student-dropout-predictor"
IMAGE_TAG="v1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print section header
section() {
    echo -e "\n${GREEN}=== $1 ===${NC}\n"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
section "Checking prerequisites"

if ! command_exists gcloud; then
    echo -e "${RED}Error: gcloud CLI not found. Please install Google Cloud SDK.${NC}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! command_exists kubectl; then
    echo -e "${RED}Error: kubectl not found. Please install kubectl.${NC}"
    echo "Visit: https://kubernetes.io/docs/tasks/tools/install-kubectl/"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}Error: docker not found. Please install Docker.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Set up GCP project
section "Setting up GCP project"
echo "Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
section "Enabling required APIs"
echo "Enabling Container Registry API..."
gcloud services enable containerregistry.googleapis.com
echo "Enabling Kubernetes Engine API..."
gcloud services enable container.googleapis.com

# Create GKE cluster
section "Creating GKE cluster"
echo "Creating cluster: $CLUSTER_NAME in $ZONE"
gcloud container clusters create $CLUSTER_NAME \
    --zone $ZONE \
    --num-nodes 2 \
    --machine-type e2-standard-2 \
    --disk-size 20

# Get credentials for kubectl
section "Getting cluster credentials"
gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE

# Build and push Docker image
section "Building and pushing Docker image"
echo "Building image: $IMAGE_NAME:$IMAGE_TAG"
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG ..

echo "Pushing image to Container Registry"
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG

# Update Kubernetes manifests with project ID
section "Updating Kubernetes manifests"
echo "Replacing PROJECT_ID in deployment.yaml"
sed -i "s/PROJECT_ID/$PROJECT_ID/g" ../k8s/deployment.yaml

# Deploy to Kubernetes
section "Deploying to Kubernetes"
echo "Applying deployment"
kubectl apply -f ../k8s/deployment.yaml
echo "Applying service"
kubectl apply -f ../k8s/service.yaml
echo "Applying HPA"
kubectl apply -f ../k8s/hpa.yaml

# Wait for deployment to be ready
section "Waiting for deployment to be ready"
kubectl rollout status deployment/student-dropout-predictor

# Get service URL
section "Getting service URL"
echo "Waiting for LoadBalancer to be assigned an external IP..."
while [ -z "$(kubectl get service student-dropout-predictor -o jsonpath='{.status.loadBalancer.ingress[0].ip}')" ]; do
    echo -n "."
    sleep 5
done
echo ""

SERVICE_IP=$(kubectl get service student-dropout-predictor -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo -e "${GREEN}Service is available at: http://$SERVICE_IP${NC}"

# Print deployment info
section "Deployment Information"
echo "Deployment status:"
kubectl get deployments
echo ""
echo "Service status:"
kubectl get services
echo ""
echo "HPA status:"
kubectl get hpa
echo ""

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo "To test the API, you can use:"
echo "curl -X POST -H \"Content-Type: application/json\" -d '{\"feature1\": value1, \"feature2\": value2, ...}' http://$SERVICE_IP/predict"
echo ""
echo "To monitor the deployment:"
echo "kubectl get pods"
echo "kubectl logs -f <pod-name>"
echo ""
echo "To clean up resources when done:"
echo "kubectl delete -f ../k8s/"
echo "gcloud container clusters delete $CLUSTER_NAME --zone $ZONE"
# Kubernetes Deployment Guide for Student Dropout Prediction API
## Complete Step-by-Step Instructions for GCP GKE Deployment

### 🎯 Overview
You'll deploy your Flask-RESTX API with Swagger UI to Google Kubernetes Engine (GKE) using the Docker image you've already uploaded to Artifact Registry.

**Your image**: `asia-southeast1-docker.pkg.dev/pnj-sp-op/student-dropout-predictor/student-dropout-predictor:v1`

---

## Step 1: Create GKE Cluster (If Not Already Done)

### Using Cloud Shell (Recommended)
In your GCP Console, click the Cloud Shell button (terminal icon) and run:

```bash
# Set your project (already done based on screenshot)
gcloud config set project pnj-sp-op

# Create a GKE cluster in the same region as your Artifact Registry
gcloud container clusters create student-dropout-cluster \
  --zone=asia-southeast1-a \
  --num-nodes=2 \
  --machine-type=e2-standard-2 \
  --disk-size=20GB \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=5
```

**What this does:**
- Creates a Kubernetes cluster with 2 nodes
- Uses cost-effective e2-standard-2 machines (2 vCPU, 8GB RAM)
- Enables auto-scaling from 1-5 nodes based on demand
- Places cluster in same region as your Docker image

⏱️ **This takes about 5-10 minutes to complete**

---

## Step 2: Configure kubectl Access

```bash
# Get credentials to access your cluster
gcloud container clusters get-credentials student-dropout-cluster \
  --zone=asia-southeast1-a \
  --project=pnj-sp-op

# Verify connection
kubectl get nodes
```

**Expected output:**
```
NAME                                                  STATUS   ROLES    AGE   VERSION
gke-student-dropout-cluster-default-pool-xxxxx       Ready    <none>   2m    v1.27.x
gke-student-dropout-cluster-default-pool-xxxxx       Ready    <none>   2m    v1.27.x
```

---

## Step 3: Deploy Your Application

### 3.1 Deploy the Main Application
```bash
# Apply the deployment (creates your pods)
kubectl apply -f k8s/deployment.yaml
```

**What this does:**
- Creates 2 replicas of your Flask-RESTX API
- Configures health checks on `/health` endpoint
- Sets resource limits (256Mi-512Mi memory, 100m-500m CPU)

### 3.2 Create the Load Balancer Service
```bash
# Apply the service (exposes your app to the internet)
kubectl apply -f k8s/service.yaml
```

**What this does:**
- Creates a Google Cloud Load Balancer
- Exposes your API on port 80
- Routes traffic to your pods on port 8080

### 3.3 Enable Auto-Scaling
```bash
# Apply horizontal pod autoscaler
kubectl apply -f k8s/hpa.yaml
```

**What this does:**
- Automatically scales pods from 1-5 based on CPU usage
- Triggers scaling when CPU > 70%

---

## Step 4: Monitor Deployment Progress

### Check Pod Status
```bash
# Watch pods starting up
kubectl get pods -w
```

**Wait for this output:**
```
NAME                                        READY   STATUS    RESTARTS   AGE
student-dropout-predictor-xxxxxxxxx-xxxxx   1/1     Running   0          2m
student-dropout-predictor-xxxxxxxxx-xxxxx   1/1     Running   0          2m
```

### Check Service and Get External IP
```bash
# Get the external IP address
kubectl get service student-dropout-predictor
```

**Wait for EXTERNAL-IP (takes 2-3 minutes):**
```
NAME                         TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)        AGE
student-dropout-predictor    LoadBalancer   10.xx.xxx.xxx   34.xxx.xxx.xxx   80:xxxxx/TCP   3m
```

⚠️ **If EXTERNAL-IP shows `<pending>`, wait a few more minutes and check again**

---

## Step 5: Test Your Deployment

### 5.1 Test Health Check
```bash
# Replace EXTERNAL-IP with your actual IP
curl http://EXTERNAL-IP/health/
```

**Expected response:**
```json
{"status": "healthy"}
```

### 5.2 Access Swagger UI Documentation
Open in your browser:
```
http://EXTERNAL-IP/docs/
```

**You should see:**
- Professional Swagger UI interface
- "Student Dropout Prediction API" title
- All 5 endpoints documented with examples
- Interactive "Try it out" buttons

### 5.3 Test Example Data Endpoint
```bash
curl http://EXTERNAL-IP/predict/example
```

**Expected response:**
```json
{
  "example_input": {
    "Admission grade": 127.3,
    "Age at enrollment": 20,
    ...
  },
  "note": "Test student data - modify values as needed for testing"
}
```

---

## Step 6: Useful Kubernetes Commands

### Viewing Logs
```bash
# View logs from all pods
kubectl logs -l app=student-dropout-predictor

# View logs from specific pod
kubectl logs student-dropout-predictor-xxxxxxxxx-xxxxx

# Follow live logs
kubectl logs -f -l app=student-dropout-predictor
```

### Scaling Your Application
```bash
# Manually scale to 3 replicas
kubectl scale deployment student-dropout-predictor --replicas=3

# Check current scale
kubectl get deployment student-dropout-predictor
```

### Viewing Resource Usage
```bash
# Check pod resource usage
kubectl top pods

# Check node resource usage
kubectl top nodes

# Check HPA status
kubectl get hpa
```

---

## Step 7: Testing Your API

### Test Single Prediction
```bash
curl -X POST http://EXTERNAL-IP/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "Admission grade": 127.3,
    "Age at enrollment": 20,
    "Application mode": 17,
    "Application order": 5,
    "Course": 171,
    "Curricular units 1st sem (approved)": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 0,
    "Curricular units 1st sem (evaluations)": 0,
    "Curricular units 1st sem (grade)": 0,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (approved)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 0,
    "Curricular units 2nd sem (evaluations)": 0,
    "Curricular units 2nd sem (grade)": 0,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Daytime/evening attendance": 1,
    "Debtor": 0,
    "Displaced": 1,
    "Educational special needs": 0,
    "Father'\''s occupation": 9,
    "Father'\''s qualification": 12,
    "GDP": 1.74,
    "Gender": 1,
    "Inflation rate": 1.4,
    "International": 0,
    "Marital status": 1,
    "Mother'\''s occupation": 5,
    "Mother'\''s qualification": 19,
    "Nacionality": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 122,
    "Scholarship holder": 0,
    "Tuition fees up to date": 1,
    "Unemployment rate": 10.8
  }'
```

---

## Step 8: Production Considerations

### Enable HTTPS (Optional but Recommended)
```bash
# Install cert-manager for automatic SSL certificates
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Set up Monitoring
```bash
# Enable GKE monitoring
gcloud container clusters update student-dropout-cluster \
  --enable-cloud-monitoring \
  --zone=asia-southeast1-a
```

### Resource Optimization
- Monitor your pods with `kubectl top pods`
- Adjust memory/CPU limits in `k8s/deployment.yaml` if needed
- Use the HPA to handle traffic spikes automatically

---

## Troubleshooting

### If Pods Won't Start
```bash
# Check pod events
kubectl describe pods

# Check if image can be pulled
kubectl get events --sort-by=.metadata.creationTimestamp
```

### If Service Shows <pending> EXTERNAL-IP
```bash
# Check service events
kubectl describe service student-dropout-predictor

# Verify firewall rules allow traffic
gcloud compute firewall-rules list
```

### If Health Checks Fail
```bash
# Check if app is listening on correct port
kubectl logs student-dropout-predictor-xxxxx

# Test internal connectivity
kubectl exec -it student-dropout-predictor-xxxxx -- curl localhost:8080/health
```

---

## Cleanup (When Done Testing)

### Delete the Application
```bash
kubectl delete -f k8s/
```

### Delete the Cluster
```bash
gcloud container clusters delete student-dropout-cluster \
  --zone=asia-southeast1-a
```

---

## Your URLs After Deployment

Once deployed with EXTERNAL-IP `34.xxx.xxx.xxx`:

- **Swagger UI**: `http://34.xxx.xxx.xxx/docs/`
- **Health Check**: `http://34.xxx.xxx.xxx/health/`
- **API Predictions**: `http://34.xxx.xxx.xxx/predict/`
- **Example Data**: `http://34.xxx.xxx.xxx/predict/example`
- **Model Metadata**: `http://34.xxx.xxx.xxx/metadata/`

🎉 **Your ML API with professional Swagger UI documentation is now running on Kubernetes in production!**
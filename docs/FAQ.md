# Frequently Asked Questions (FAQ)
## Student Dropout Prediction Project

### 📋 Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Data and Model Questions](#data-and-model-questions)
3. [API Usage and Testing](#api-usage-and-testing)
4. [Deployment and Infrastructure](#deployment-and-infrastructure)
5. [Troubleshooting](#troubleshooting)
6. [Technical Architecture](#technical-architecture)
7. [Performance and Benchmarks](#performance-and-benchmarks)

---

## Setup and Installation

### Q: What are the system requirements?
**A:** 
- Python 3.9 or higher
- At least 4GB RAM (8GB recommended for model training)
- 2GB free disk space
- Docker (optional, for containerization)
- Google Cloud SDK (for cloud deployment)

### Q: How do I set up the virtual environment?
**A:** 
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Q: The dataset download fails. What should I do?
**A:** 
1. Ensure you have Kaggle API credentials configured:
   ```bash
   # Create ~/.kaggle/kaggle.json with your API key
   {"username":"your_username","key":"your_api_key"}
   ```
2. Set proper permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Alternative: Download manually from Kaggle and place in `data/` folder

### Q: I get import errors when running the notebook. How to fix?
**A:** 
1. Ensure virtual environment is activated
2. Install Jupyter in the virtual environment: `pip install jupyter`
3. Register the environment as a kernel:
   ```bash
   python -m ipykernel install --user --name=student-dropout --display-name="Student Dropout"
   ```
4. Select the correct kernel in Jupyter

---

## Data and Model Questions

### Q: What does the dataset contain?
**A:** 
The dataset contains 4,424 student records with:
- **35 features** including demographics, academic history, and socio-economic factors
- **3 target classes**: Dropout, Enrolled, Graduate
- **Key features**: Admission grade, curricular units performance, family background, economic indicators

### Q: Why did you choose these specific models?
**A:** 
Our model selection strategy:
1. **Logistic Regression**: Simple baseline for interpretability
2. **Random Forest**: Handles mixed data types, provides feature importance
3. **Gradient Boosting**: Sequential learning, often performs well on tabular data
4. **XGBoost**: State-of-the-art gradient boosting with advanced regularization

### Q: What is the best performing model and why?
**A:** 
**Tuned Gradient Boosting** achieved the best performance:
- **Accuracy**: 78.04%
- **F1-Score**: 0.7756 (macro-averaged)
- **Advantages**: Balanced performance across all classes, good generalization

### Q: How do you handle class imbalance?
**A:** 
1. **Stratified sampling** in train/test splits
2. **Stratified K-fold** cross-validation
3. **Macro-averaged** F1-score for evaluation
4. **Class weight balancing** in some models

### Q: What features are most important for prediction?
**A:** 
Top predictive features (from XGBoost analysis):
1. Curricular units 1st/2nd semester performance
2. Age at enrollment
3. Previous qualification grade
4. Admission grade
5. Economic indicators (GDP, unemployment rate)

---

## API Usage and Testing

### Q: What API endpoints are available?
**A:** 
- **GET /health**: Service health check
- **GET /docs**: Interactive Swagger UI documentation
- **GET /metadata**: Model information and performance metrics
- **GET /predict/example**: Sample input data format
- **POST /predict**: Single student prediction
- **POST /predict/batch**: Multiple student predictions

### Q: How do I use the Swagger UI documentation?
**A:** 
1. Start the service: `python scripts/predict.py`
2. Open browser to: `http://localhost:8080/docs/`
3. Click "Try it out" on any endpoint
4. Fill in example data and execute
5. View real-time responses and schema validation

### Q: What's the expected input format for predictions?
**A:** 
```json
{
  "Admission grade": 127.3,
  "Age at enrollment": 20,
  "Application mode": 17,
  "Course": 171,
  "Curricular units 1st sem (approved)": 0,
  "Curricular units 1st sem (grade)": 0.0,
  "GDP": 1.74,
  "Gender": 1,
  "Marital status": 1,
  ... // 35 total features
}
```
Use `/predict/example` endpoint to get complete sample data.

### Q: How do I test the API programmatically?
**A:** 
```bash
# Health check
curl http://localhost:8080/health/

# Get example data
curl http://localhost:8080/predict/example

# Make prediction
curl -X POST http://localhost:8080/predict/ \
  -H "Content-Type: application/json" \
  -d @sample_data.json

# Run comprehensive tests
python tests/test_prediction.py --url http://localhost:8080
```

### Q: What do the prediction responses mean?
**A:** 
```json
{
  "prediction": "Graduate",
  "probabilities": {
    "Dropout": 0.15,
    "Enrolled": 0.25,
    "Graduate": 0.60
  },
  "timestamp": "2024-01-20T08:00:00"
}
```
- **prediction**: Most likely outcome
- **probabilities**: Confidence scores for each class (sum to 1.0)
- **timestamp**: When prediction was made

---

## Deployment and Infrastructure

### Q: How do I deploy to Docker locally?
**A:** 
```bash
# Build image
docker build -t student-dropout-predictor:v1 .

# Run container
docker run -p 8080:8080 student-dropout-predictor:v1

# Test
curl http://localhost:8080/health/
```

### Q: How do I deploy to Google Cloud?
**A:** 
See detailed instructions in [`docs/kubernetes_deployment_guide.md`](kubernetes_deployment_guide.md).

Quick steps:
1. Build and push to Artifact Registry
2. Create GKE cluster
3. Deploy with `kubectl apply -f k8s/`
4. Get external IP and test

### Q: What cloud resources does this use?
**A:** 
- **Google Kubernetes Engine (GKE)**: Container orchestration
- **Artifact Registry**: Docker image storage
- **Load Balancer**: Traffic distribution
- **Horizontal Pod Autoscaler**: Auto-scaling
- **Cloud Monitoring**: Logging and metrics

### Q: How much does cloud deployment cost?
**A:** 
Estimated monthly costs (asia-southeast1):
- **GKE cluster** (2 x e2-standard-2): ~$50-70/month
- **Load Balancer**: ~$20/month
- **Artifact Registry**: ~$1-5/month
- **Total**: ~$70-95/month

Use `gcloud compute instances stop` when not needed to save costs.

### Q: Can I use other cloud providers?
**A:** 
Yes! The Docker image works on:
- **AWS**: EKS, ECS, or EC2
- **Azure**: AKS or Container Instances
- **Any Kubernetes**: Modify k8s/ files as needed

---

## Troubleshooting

### Q: The prediction service won't start. What should I check?
**A:** 
1. **Check model files exist**:
   ```bash
   ls -la models/
   # Should see: best_model_gradient_boosting.joblib, label_encoder.joblib
   ```

2. **Verify dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check port availability**:
   ```bash
   # Windows
   netstat -an | findstr :8080
   # Linux/Mac
   lsof -i :8080
   ```

4. **View detailed logs**:
   ```bash
   python scripts/predict.py
   # Check logs/prediction.log
   ```

### Q: Model loading fails with "FileNotFoundError"
**A:** 
1. **Train the model first**:
   ```bash
   python scripts/train.py
   ```

2. **Check model path**:
   ```bash
   export MODEL_PATH=./models/best_model_gradient_boosting.joblib
   python scripts/predict.py
   ```

3. **Verify file permissions**:
   ```bash
   chmod 644 models/*.joblib
   ```

### Q: Docker build fails
**A:** 
Common issues:
1. **Large image size**: Use `.dockerignore` to exclude unnecessary files
2. **Network timeouts**: Use `--network=host` or configure proxy
3. **Permission denied**: Run as administrator/sudo
4. **Cache issues**: Use `docker build --no-cache`

### Q: Kubernetes deployment shows "ImagePullBackOff"
**A:** 
1. **Check image exists**:
   ```bash
   gcloud container images list --repository=asia-southeast1-docker.pkg.dev/PROJECT_ID/student-dropout-predictor
   ```

2. **Configure authentication**:
   ```bash
   gcloud auth configure-docker asia-southeast1-docker.pkg.dev
   ```

3. **Verify image path in deployment.yaml**

4. **Check pod events**:
   ```bash
   kubectl describe pods
   ```

### Q: API returns 500 Internal Server Error
**A:** 
1. **Check input data format**: Ensure all 35 features are present
2. **Validate data types**: Numeric values should be numbers, not strings
3. **Check logs**: `kubectl logs deployment/student-dropout-predictor`
4. **Test locally first**: Verify API works locally before cloud deployment

### Q: Swagger UI shows "Failed to fetch"
**A:** 
1. **Check CORS settings**: Service should allow browser requests
2. **Verify URL**: Ensure `/docs/` endpoint is accessible
3. **Clear browser cache**: Hard refresh (Ctrl+F5)
4. **Check network**: Ensure no firewall blocks requests

---

## Technical Architecture

### Q: How is the project structured?
**A:** 
```
Project Structure:
├── data/           # Dataset and visualizations
├── notebook/       # Jupyter notebooks for analysis
├── models/         # Trained models and metadata
├── scripts/        # Training and prediction scripts
├── k8s/           # Kubernetes deployment configs
├── docs/          # Documentation
└── tests/         # Testing scripts
```

### Q: What machine learning pipeline is used?
**A:** 
1. **Data Preprocessing**: StandardScaler for numerical, OneHotEncoder for categorical
2. **Cross-validation**: 5-fold stratified for robust evaluation
3. **Hyperparameter Tuning**: GridSearchCV and Optuna optimization
4. **Model Selection**: Best model based on macro-averaged F1-score
5. **Production Pipeline**: Scikit-learn Pipeline with preprocessing and model

### Q: How is the Flask-RESTX API structured?
**A:** 
- **Namespaces**: Organized endpoints (`/health`, `/predict`, `/metadata`)
- **Data Models**: Pydantic-style validation for inputs/outputs
- **Auto-documentation**: Swagger UI with interactive testing
- **Error Handling**: Standardized error responses
- **Logging**: Comprehensive request/response logging

### Q: What's the difference between this and FastAPI?
**A:** 
We chose **Flask-RESTX** over FastAPI migration:
- ✅ **Minimal code changes**: Enhanced existing Flask app
- ✅ **Same functionality**: Auto-generated Swagger UI docs
- ✅ **Proven reliability**: Mature Flask ecosystem
- ✅ **Easy deployment**: No breaking changes to Docker/K8s

### Q: How does auto-scaling work?
**A:** 
**Horizontal Pod Autoscaler (HPA)**:
- **Triggers**: CPU usage > 70%
- **Range**: 1-5 pods
- **Metrics**: CPU and memory utilization
- **Response time**: ~30 seconds to scale up/down

---

## Performance and Benchmarks

### Q: What are the model performance metrics?
**A:** 

| Model | Accuracy | F1-Score | Training Time | Inference Speed |
|-------|----------|----------|---------------|-----------------|
| Logistic Regression | 74.2% | 0.7156 | 2s | ~1ms |
| Random Forest | 76.8% | 0.7534 | 15s | ~5ms |
| **Gradient Boosting** | **78.0%** | **0.7756** | **45s** | **~3ms** |
| XGBoost | 77.1% | 0.7612 | 35s | ~4ms |

### Q: How fast is the API?
**A:** 
**Performance Benchmarks**:
- **Single prediction**: ~50ms (including network)
- **Batch prediction** (10 students): ~80ms
- **Model loading time**: ~2 seconds
- **Memory usage**: ~200MB per pod
- **Throughput**: ~500 requests/minute per pod

### Q: What's the business impact?
**A:** 
**ROI Analysis** (see [`docs/business_impact.md`](business_impact.md)):
- **Cost savings**: $2.5M annually (1000-student institution)
- **ROI**: 312% in first year
- **Break-even**: 4.2 months
- **Intervention success**: 65% retention improvement

### Q: How does it compare to other student retention solutions?
**A:** 
**Competitive Advantages**:
- ✅ **Early prediction**: Works with enrollment data only
- ✅ **Multi-class output**: Dropout/Enrolled/Graduate predictions
- ✅ **High accuracy**: 78% vs industry average 60-70%
- ✅ **Cost-effective**: Open source, cloud-native
- ✅ **Interpretable**: Clear feature importance rankings

### Q: What are the model limitations?
**A:** 
**Known Limitations**:
1. **Data dependency**: Requires consistent data quality
2. **Temporal drift**: Model may need retraining over time
3. **Feature scope**: Limited to enrollment-time features
4. **Class imbalance**: Slightly favors majority class (Graduate)
5. **Interpretability**: Ensemble methods less interpretable than linear models

### Q: How often should the model be retrained?
**A:** 
**Retraining Recommendations**:
- **Minimum**: Annually with new student cohort data
- **Optimal**: Bi-annually for concept drift detection
- **Trigger-based**: When accuracy drops below 75%
- **Data-driven**: When feature distributions change significantly

---

## 🚀 Getting Started Checklist

New to the project? Follow this checklist:

- [ ] Clone repository and set up virtual environment
- [ ] Download dataset with `python scripts/download_data.py`
- [ ] Run Jupyter notebook for exploration
- [ ] Train model with `python scripts/train.py`
- [ ] Test API locally with `python scripts/predict.py`
- [ ] Access Swagger UI at `http://localhost:8080/docs/`
- [ ] Run tests with `python tests/test_prediction.py`
- [ ] Build Docker image and test containerization
- [ ] Deploy to cloud following deployment guide

---

## 📞 Support and Contributing

- **Issues**: Create GitHub issue with error logs
- **Feature requests**: Submit detailed use case description
- **Contributing**: Follow standard Python/ML best practices
- **Documentation**: All changes should update relevant docs

**Contact**: See README.md for maintainer information

---

*This FAQ covers common questions based on user feedback and deployment experiences. For technical implementation details, see the source code and inline documentation.*
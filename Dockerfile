FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and scripts
COPY models/ /app/models/
COPY scripts/ /app/scripts/

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV MODEL_PATH=/app/models/model.joblib
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the prediction service
CMD ["python", "scripts/predict.py"]
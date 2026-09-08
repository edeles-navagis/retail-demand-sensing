FROM python:3.10-slim

WORKDIR /usr/src/app

# Install system dependencies (LightGBM requires libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and code
COPY demand_prediction_model.pkl ./
COPY label_encoders.pkl ./
COPY feature_columns.pkl ./
COPY serve.py ./

# Expose default port (Vertex AI injects AIP_HTTP_PORT, but defaults to 8080)
EXPOSE 8080

# Command to launch the server
CMD ["python", "serve.py"]

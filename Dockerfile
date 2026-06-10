# Use official Python image with CUDA support if GPU is needed, else use standard python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install build tools for scientific packages
RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY new_way/ new_way/
COPY scripts/ scripts/
COPY logging_config.ini ./
COPY models/ models/
COPY data/ data/
COPY results/ results/
COPY new_way/config.yaml ./config.yaml

# Set environment variable for CUDA (optional, can be overridden)
ENV CUDA_VISIBLE_DEVICES=0

# Default command to run the inference script
CMD ["python", "scripts/test_inference.py"]

# Use Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Create necessary directories with proper permissions
RUN mkdir -p logs data/queries data/feedback \
    && chmod 755 logs data/queries data/feedback

# Copy the application code
COPY . .

# Set environment variables
ENV PORT=8080

# Use Gunicorn with Uvicorn worker for ASGI support
CMD exec gunicorn \
    --bind :$PORT \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --threads 8 \
    --timeout 0 \
    main:app

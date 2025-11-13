# Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python + essentials
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy model (you'll mount it later)
# We'll mount /app/model at runtime

# Copy app
COPY app.py .
COPY requirements.txt .

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5023

# Run
CMD ["python3", "app.py"]

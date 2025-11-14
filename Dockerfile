# Dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and SoX
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py .
COPY requirements.txt .

# Install numpy first (prevents Nemo build error)
RUN pip3 install --no-cache-dir numpy==1.26.4

# Install all dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5023

CMD ["python3", "app.py"]
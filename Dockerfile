FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git ffmpeg libsndfile1 sox \
    build-essential cython3 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Working directory
# -------------------------------
WORKDIR /app

# Copy app and requirements
COPY app.py .
COPY requirements.txt .

# -------------------------------
# Preinstall fixed versions for numpy + typing_extensions
# (fixes many pip conflicts)
# -------------------------------
RUN pip3 install --no-cache-dir numpy==1.23.5 typing_extensions==4.10.0

# -------------------------------
# Install the required packages
# -------------------------------
RUN pip3 install --no-cache-dir -r requirements.txt

# -------------------------------
# Expose port
# -------------------------------
EXPOSE 5023

# -------------------------------
# Entry point
# -------------------------------
CMD ["python3", "app.py"]
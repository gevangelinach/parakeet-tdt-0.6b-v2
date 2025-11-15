FROM nvcr.io/nvidia/nemo:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 sox \
    build-essential cython3 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Working directory
# -------------------------------
WORKDIR /app

COPY app.py .
COPY requirements.txt .

# -------------------------------
# Install your extra dependencies
# (NeMo + Pytorch already installed)
# -------------------------------
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5023

CMD ["python3", "app.py"]

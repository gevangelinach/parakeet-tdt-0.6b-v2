# Use a valid NeMo container version — try 25.04
FROM nvcr.io/nvidia/nemo:25.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 sox \
    build-essential cython3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py .
COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5023

CMD ["python3", "app.py"]

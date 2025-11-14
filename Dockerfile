FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git ffmpeg libsndfile1 sox \
    build-essential cython3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py .
COPY requirements.txt .

RUN pip3 install --no-cache-dir numpy==1.23.5 typing_extensions==4.10.0

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5023

CMD ["python3", "app.py"]
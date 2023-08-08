FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git

COPY src /app/src
WORKDIR /app

RUN pip install -r requirements.txt

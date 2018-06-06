# Dockerfile for the detector processing loop.

# Using nvidia-cuda as the base image
FROM nvidia/cuda:9.1-cudnn7-devel

# Copy contents of adminweb into the container's /opt folder
COPY . /opt/feed_ingestor
WORKDIR /opt/feed_ingestor

# Environment variables to make Python use UTF-8 as the default encoding
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# Install Tools
RUN \
  apt-get update && \
  apt-get install -y \
    cmake \
    gcc \
    libgtk2.0-dev \
    python3 \
    python3-pip

# Install all Python requirements for the processing loop
RUN pip3 install -r requirements.txt

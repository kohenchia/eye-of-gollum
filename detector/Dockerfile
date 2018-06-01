# Dockerfile for the detector processing loop.

# Using nvidia-cuda as the base image
FROM nvidia/cuda:9.1-cudnn7-devel

# Install Tools
RUN \
  apt-get update && \
  apt-get install -y python3 python3-pip gcc

# Install all Python requirements for the processing loop
RUN pip3 install -r requirements.txt

# Copy contents of adminweb into the container's /opt folder
COPY . /opt/detector
WORKDIR /opt/detector
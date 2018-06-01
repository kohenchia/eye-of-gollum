# Dockerfile for the web interface.
# The web interface is served by an aiohttp web server.

# Use python:3.6.5-alpine as the base image
FROM python:3.6.5-slim-jessie

# Copy contents of adminweb into the container's /opt folder
COPY . /opt/videoserver
WORKDIR /opt/videoserver

# Install GCC, required by pip to install aioredis
RUN apt-get update && apt-get install -y gcc

# Install all Python requirements for the web server
RUN pip install -r requirements.txt
# eye-of-gollum: A Prototype Facial Badge System

## Overview

This repository contains a simple facial badge system that runs facial recognition on a video feed in real time and displays results through a web interface.

## Architecture
This system is made up of four main components:

1. A **detector** module that runs a processing loop to ingest frames from an external video feed and perform facial recognition on them. Annotated bounding boxes are added to each frame and then saved in a Redis cache.

2. A **Redis cache** used to store annotated frames in memory.

3. A **video streaming server** that continuously streams the annotated frames from the Redis cache in Motion-JPEG format over an HTTP endpoint.

4. A simple React-based **web interface** to display the results from the video streaming server.

## System Requirements

1. A CUDA-enabled GPU
2. The NVIDIA container runtime for Docker ([`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)) installed on your machine

## Running the System

The Redis cache, detector, and video server components are implemented as Docker containers. Deployment is managed through `docker-compose`. To start the system, run:

```
$ docker-compose up -d
```

This will start three Docker containers:

1. `eog-rediscache` (based on the [`redis:alpine`](https://hub.docker.com/_/redis/) image) to host the Redis cache
2. `eog-detector` (built from `/detector`) to host the detector module on 
3. `eog-videoserver` (built from `/videoserver`) to host the video streaming server

To start each container independently, run:

```
$ docker-compose up -d redis
$ docker-compose up -d detector
$ docker-compose up -d videoserver
```

Port configurations for all containers can be found in `docker-compose.yaml`.

The React-based web interface should be hosted as static files directly behind a web server like [NGINX](https://www.nginx.com/). More detailed documentation is available in the README file in the `/web` folder.

## Deploying to Production

Don't. This is a prototype designed to run on a single machine.

To convert this into a production-ready application, you will at least want to break up the components into their own repositories, deploy them in a distributed architecture using an orchestration framework like [Kubernetes](https://kubernetes.io/), and host the web application through a CDN like [AWS CloudFront](https://aws.amazon.com/cloudfront/)
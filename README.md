# eye-of-gollum: A Prototype Facial Badge System

## Overview

This repository contains a simple facial badge system that runs facial recognition on a video feed in real time and displays results through a web interface.

## Architecture
This system is made up of five main components:

1. A **Redis cache** that stores raw and processed images in memory.

2. A **feed ingestor** module that runs a processing loop to ingest frames from an external video feed and stores them in the cache.

3. An **annotator** module that performs facial recognition on the raw video frames. Annotated bounding boxes are added to each frame and then saved in the cache.

4. A **video streaming server** that continuously streams the annotated frames from the cache through a WebSockets connection.

5. A simple React-based **web interface** to display the results from the video streaming server.

## System Requirements

1. A CUDA-enabled GPU
2. The NVIDIA container runtime for Docker ([`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker)) installed on your machine

## Running the System

The Redis cache, frame ingestor, annotator, and video server components are implemented as Docker containers. Deployment is managed through `docker-compose`. To start the system, run:

```
$ docker-compose up -d
```

This will start four Docker containers:

1. `eog-rediscache` (based on the [`redis:alpine`](https://hub.docker.com/_/redis/) image) to host the Redis cache
2. `eog-feed-ingestor-001` (built from `/feed_ingestor`) to host the feed ingestion module
3. `eog-annotator-001` (built from `/annotator`) to host the annotator module
4. `eog-video-server-001` (built from `/video_server`) to host the video streaming server

To start each container independently, run:

```
$ docker-compose up -d redis
$ docker-compose up -d feed_ingestor_001
$ docker-compose up -d annotator_001
$ docker-compose up -d video_server_001
```

All containers ending with `_001` are horizontally scalable, and multiple instances can be deployed as long as the port configurations are managed properly. In addition to scaling benefits, this architecture also enables multiple types of feeds and annotators to run at the same time for comparison.

Port configurations for all containers can be found in `docker-compose.yaml`.

The React-based web interface should be hosted as static files directly behind a web server like [NGINX](https://www.nginx.com/). More detailed documentation is available in the README file in the `/web` folder.

## Deploying to Production

Don't. This is a prototype designed to run on a single machine.

To convert this into a production-ready application, you will at least want to break up the components into their own repositories, deploy them in a distributed architecture using an orchestration framework like [Kubernetes](https://kubernetes.io/), and host the web application through a CDN like [AWS CloudFront](https://aws.amazon.com/cloudfront/)
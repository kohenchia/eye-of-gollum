# Eye of Gollum: A Prototype Facial Badge System

## Overview

This repository contains a simple facial badge system that runs facial recognition on a video feed in real time and displays results through a web interface.

## Architecture
This system is made up of four main components:

1. A **detector** module that runs a processing loop to ingest frames from an external video feed and perform facial recognition on them. Annotated bounding boxes are added to each frame and then saved in a Redis cache.

2. A **Redis cache** used to store annotated frames in memory.

3. A **video streaming server** that continuously streams the annotated frames from the Redis cache in Motion-JPEG format over an HTTP endpoint.

4. A simple React-based **web interface** to display the results from the video streaming server.

## Running the System

System deployment is managed through `docker-compose`. To start the system, run:

```
$ docker-compose up -d
```

This will start three Docker containers for the first three components:

1. `ai2incubator/eog-rediscache` (based on the [`redis:alpine`](https://hub.docker.com/_/redis/) image) to host the Redis cache
2. `ai2incubator/eog-detector` (built from `detector.Dockerfile`) to host the detector module on 
3. `ai2incubator/eog-videoserver` (built from `videoserver.Dockerfile`) to host the video streaming server

The React web interface should be hosted as static files directly behind a web server like [NGINX](https://www.nginx.com/) or through a CDN. For convenience, `docker-compose.yaml` includes a `dev` mode that additionally runs a fourth Docker container that serves the web interface locally for development purposes. To start the system in `dev` mode, run:

(Fix)
```
$ docker-compose up -d --dev
```

Alternatively, you can also host the web interface through your local web server by mapping the `web` folder to a local port, or through a development Node.js / webpack server if you are actively developing on it.

Port configurations for all four containers can be found in `docker-compose.yaml`.

Contact: kohenchia@gmail.com
# fbs-deploy

Deployment script for the Facial Badge System.

System deployment is managed through Docker compose. It requires three images:

1. [`redis:alpine`](https://hub.docker.com/_/redis/)
2. [`fbs-detector`](https://bitbucket.org/kohenchia-ai2/fbs-detector/src)
3. [`fbs-adminweb`](https://bitbucket.org/kohenchia-ai2/fbs-adminweb/src)

The Redis container is used as an in-memory cache to store processed frames.

The `fbs-detector` container runs a processing loop to ingest frames from a video feed and run facial recognition on them. It saves the processed frames into the Redis cache.

The `fbs-adminweb` container hosts a web server that serves the admin UI.

To start the system, run:
```
$ docker-compose up
```
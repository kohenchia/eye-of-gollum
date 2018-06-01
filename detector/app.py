import time
import logging
import argparse

# Project dependencies
import cv2
import redis
import face_recognition
from graceful_killer import GracefulKiller

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def start_loop(redis_host='localhost', redis_port=6379):
    """
    Starts the processing loop
    """
    frame = 0
    cache = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
    killer = GracefulKiller()

    while True:
        # TODO: Grab frame from webcam
        # TODO: Run face detection on webcam
        # TODO: Add bounding box to frame
        # TODO: Store the processed frame in Redis
        cache.set('frame', frame)
        frame += 1

        # Terminal 
        if killer.kill_now:
            LOG.info('eog-detector: Shutting down gracefully...')
            break

        time.sleep(1)


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--redis_host', required=True)
    parser.add_argument('--redis_port', required=True)
    args = parser.parse_args()

    start_loop(redis_host=args.redis_host, 
               redis_port=args.redis_port)

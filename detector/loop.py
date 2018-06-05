"""Processing loop that pulls frames from a video feed and runs facial recognition on them."""
import io
import time
import base64
import logging
import argparse

# Project dependencies
import cv2
import redis
import face_recognition

from PIL import Image
from graceful_killer import GracefulKiller
from ipcam import IPCam
from annotator import Annotator

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%H:%M:%S')
LOG = logging.getLogger(__name__)


def start_loop(stream_url=None, redis_host='localhost', redis_port=6379):
    """
    Starts the processing loop
    """
    killer = GracefulKiller()

    LOG.info('Initializing Redis cache...')
    cache = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    LOG.info('Initializing IP Camera...')
    camera = IPCam(stream_url=stream_url)

    LOG.info('Initializing frame annotator...')
    annotator = Annotator()

    LOG.info('Starting processing loop...')
    while True:
        # Get next frame and annotate with bounding boxes
        frame = camera.get_next_frame()
        frame = annotator.annotate(frame)

        # Serialize frame to base64
        image = Image.fromarray(frame)        
        raw_bytes = io.BytesIO()
        image.save(raw_bytes, format='JPEG')
        image_b64 = str(base64.b64encode(raw_bytes.getvalue()))

        # Save base64-serialized frame to cache
        cache.set('frame', image_b64)

        # Capture kill signals and terminate loop
        if killer.kill_now:
            LOG.info('Shutting down gracefully...')
            break


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_url', required=True)
    parser.add_argument('--redis_host', required=True)
    parser.add_argument('--redis_port', required=True)
    args = parser.parse_args()

    start_loop(stream_url=args.stream_url,
               redis_host=args.redis_host, 
               redis_port=args.redis_port)

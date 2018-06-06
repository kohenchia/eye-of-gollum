"""Processing loop that pulls frames from a video feed and runs facial recognition on them."""
import argparse
import logging
import redis
from PIL import Image

# Local dependencies
from annotator import Annotator
from graceful_killer import GracefulKiller

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%H:%M:%S')
LOG = logging.getLogger(__name__)


def start_loop(stream_name=None, redis_host='localhost', redis_port=6379):
    """
    Starts the processing loop
    """
    killer = GracefulKiller()

    LOG.info('Initializing Redis cache...')
    cache = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    LOG.info('Initializing frame annotator...')
    annotator = Annotator()

    LOG.info('Starting processing loop...')
    while True:
        # Get next available frame from the cache
        image_bytes = cache.get('{}_raw'.format(stream_name))
        if image_bytes is None:
            continue

        # Annotate with bounding boxes
        # TODO: Get size from configs
        image = Image.frombytes('RGB', (1920, 1080), image_bytes)
        image = annotator.annotate(image)

        # Save frame to cache as raw bytes
        cache.set('{}_annotated'.format(stream_name), image.tobytes())

        # Capture kill signals and terminate loop
        if killer.kill_now:
            LOG.info('Shutting down gracefully...')
            break


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_name', required=True)
    parser.add_argument('--redis_host', required=True)
    parser.add_argument('--redis_port', required=True)
    args = parser.parse_args()

    start_loop(stream_name=args.stream_name,
               redis_host=args.redis_host, 
               redis_port=args.redis_port)

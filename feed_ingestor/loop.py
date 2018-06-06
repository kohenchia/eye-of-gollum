"""Processing loop that pulls frames from a video feed and runs facial recognition on them."""
import argparse
import logging
import redis
from PIL import Image

# Local dependencies
from graceful_killer import GracefulKiller
from ipcam import IPCam

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%H:%M:%S')
LOG = logging.getLogger('feed_ingestor')


def start_loop(stream_name=None, stream_url=None, redis_host='localhost', redis_port=6379):
    """
    Starts the processing loop
    """
    killer = GracefulKiller()

    LOG.info('Initializing Redis cache...')
    cache = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

    LOG.info('Initializing IP Camera...')
    with IPCam(stream_url=stream_url) as camera:

        LOG.info('Starting processing loop...')
        while True:
            # Get next frame and store it in Redis
            frame = camera.get_next_frame()
            if frame is None:
                continue

            # Save frame to cache as raw bytes
            image = Image.fromarray(frame)        
            cache.set('{}_raw'.format(stream_name), image.tobytes())

            # Capture kill signals and terminate loop
            if killer.kill_now:
                LOG.info('Shutting down gracefully...')
                break


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_name', required=True)
    parser.add_argument('--stream_url', required=True)
    parser.add_argument('--redis_host', required=True)
    parser.add_argument('--redis_port', required=True)
    args = parser.parse_args()

    start_loop(stream_name=args.stream_name,
               stream_url=args.stream_url,
               redis_host=args.redis_host, 
               redis_port=args.redis_port)

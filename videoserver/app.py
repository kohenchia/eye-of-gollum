# -*- coding: utf-8 -*-
"""Video server entry point written as an aiohttp application.

To start the server, run:

    $ python app.py

Alternatively, run an aiohttp development server:
    
    $ python -m aiohttp.web -H localhost -P 8080 app:start_server

"""
import argparse
import aioredis
from aiohttp import web, log
from routes import routes as main_routes


def init_server(redis_host='localhost', redis_port=6379):
    """
    Initializes the application server.
    """
    log.server_logger.info('eog-videoserver: Initializing server...')

    async def init_redis(app):
        """
        Initializes a Redis connection
        """
        log.server_logger.info('eog-videoserver: Initializing Redis connection...')
        app['redis'] = await aioredis.create_redis(
            (redis_host, redis_port),
            encoding='utf-8'
        )

    async def close_redis(app):
        """
        Shut down Redis connection
        """
        log.server_logger.info('eog-videoserver: Shutting down Redis connection...')
        app['redis'].close()

    app = web.Application()
    app.on_startup.append(init_redis)
    app.on_shutdown.append(close_redis)
    app.add_routes(main_routes)
    return app


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--redis_host', required=True)
    parser.add_argument('--redis_port', required=True)
    parser.add_argument('--port', required=True)
    args = parser.parse_args()

    # Initialize and run the server
    app = init_server(args.redis_host, args.redis_port)
    web.run_app(app, port=args.port)
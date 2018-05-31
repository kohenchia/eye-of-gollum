# -*- coding: utf-8 -*-
"""Video server entry point written as an aiohttp application.

To start the server, run:

    $ python app.py

Alternatively, run an aiohttp development server:
    
    $ python -m aiohttp.web -H localhost -P 8080 app:start_server

"""
from aiohttp import web

routes = web.RouteTableDef()


@routes.get('/')
async def main(request):
    """
    Returns a simple 200 OK response.
    Can be used as a simple health check endpoint.
    """
    return web.json_response(data={
        'status': 200
    })


@routes.get('/stream')
async def get_next_frame(request):
    """
    Returns the next frame in an x/multipart response.
    """
    return web.Response(text=3)


def init_server(argv):
    """
    Initializes the application server.
    """
    app = web.Application()
    app.add_routes(routes)
    return app


if __name__ == '__main__':
    import sys
    app = init_server(sys.argv)
    web.run_app(app, port=8777)  # TODO: Config
# -*- coding: utf-8 -*-
"""Video server endpoints"""
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
    val = await request.app['redis'].get('frame')
    return web.json_response(data={
        'frame': val
    })
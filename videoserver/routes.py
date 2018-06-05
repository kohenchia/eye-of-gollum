# -*- coding: utf-8 -*-
"""Video server endpoints"""
import base64
import logging

from aiohttp import web

routes = web.RouteTableDef()
LOG = logging.getLogger(__name__)


@routes.get('/')
async def main(request):
    """
    Returns a simple 200 OK response.
    Can be used as a simple health check endpoint.
    """
    return web.json_response(data={
        'status': 200
    })


@routes.get('/frame')
async def frame_handler(request):
    """
    Returns the next frame in an x/multipart response.
    """
    frame_bytes = await request.app['redis'].get('frame')

    # Serialize frame bytes to base64
    if frame_bytes is not None:
        frame = base64.b64encode(frame_bytes)
        frame = frame.decode('utf-8')

    return web.json_response(data={
        'frame': frame
    })


@routes.get('/stream')
async def websockets_stream_handler(request):
    """
    Returns a WebSockets response to continuously stream frames.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                await ws.close()
            else:
                frame = request.app['redis'].get('frame')
                await ws.send_bytes(frame)
        elif msg.type == aiohttp.WSMsgType.ERROR:
            LOG.error('WebSockets connection closed with exception %s' %
                  ws.exception())

    LOG.info('WebSockets connection closed')
    return ws
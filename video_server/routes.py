# -*- coding: utf-8 -*-
"""Video server endpoints"""
import aiohttp
import base64
import io
import logging
from PIL import Image

from aiohttp import web, log

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
    # TODO: Parse request args
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                await ws.close()
            else:
                # TODO: Configure stream source and size
                image_bytes = await request.app['redis'].get('001_annotated')
                image = Image.frombytes('RGB', (1920, 1080), image_bytes)

                # Resize image and serialize to JPEG
                image = image.resize((960, 540))
                jpeg_bytes = io.BytesIO()
                image.save(jpeg_bytes, format='JPEG')

                # Return bytes to client
                await ws.send_bytes(jpeg_bytes.getvalue())
        elif msg.type == aiohttp.WSMsgType.ERROR:
            log.server_logger.error('WebSockets connection closed with exception %s' %
                  ws.exception())

    log.server_logger.info('WebSockets connection closed')
    return ws

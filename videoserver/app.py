from aiohttp import web


async def main(request):
    return web.Response(text='Hi')


async def get_next_frame(request):
    return web.Response(text=3)


app = web.Application()
app.router.add_get('/', main)
app.router.add_get('/frame', get_next_frame)
web.run_app(app)
# Python IO操作 - 高级网络IO示例

import socket
import ssl
import asyncio
import aiohttp
import websockets
import json
from aiohttp import web

# 1. SSL/TLS加密通信
def secure_socket_communication():
    print("SSL/TLS加密通信示例：")
    
    # 创建SSL上下文
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile="server.crt", keyfile="server.key")
    
    # SSL服务器
    def ssl_server():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('localhost', 8443))
        server.listen(1)
        
        # 包装socket
        ssl_sock = context.wrap_socket(server, server_side=True)
        return ssl_sock
    
    # SSL客户端
    def ssl_client():
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ssl_client = context.wrap_socket(client,
                                       server_hostname='localhost')
        ssl_client.connect(('localhost', 8443))
        return ssl_client

# 2. WebSocket服务器
async def websocket_server():
    print("\nWebSocket服务器示例：")
    
    async def handler(websocket, path):
        try:
            async for message in websocket:
                print(f"收到消息: {message}")
                # 发送响应
                response = f"服务器收到: {message}"
                await websocket.send(response)
        except websockets.exceptions.ConnectionClosed:
            print("客户端断开连接")
    
    server = await websockets.serve(handler, 'localhost', 8765)
    await server.wait_closed()

# 3. REST API服务器
async def rest_api_server():
    print("\nREST API服务器示例：")
    
    # 创建应用
    app = web.Application()
    
    # 数据存储
    items = []
    
    # 路由处理器
    async def get_items(request):
        return web.json_response(items)
    
    async def add_item(request):
        data = await request.json()
        items.append(data)
        return web.json_response(data)
    
    async def get_item(request):
        id = int(request.match_info['id'])
        if 0 <= id < len(items):
            return web.json_response(items[id])
        return web.json_response({"error": "Not found"}, status=404)
    
    # 设置路由
    app.router.add_get('/api/items', get_items)
    app.router.add_post('/api/items', add_item)
    app.router.add_get('/api/items/{id}', get_item)
    
    return app

# 4. 异步文件上传
async def async_file_upload():
    print("\n异步文件上传示例：")
    
    async def upload_handler(request):
        reader = await request.multipart()
        
        # 处理文件字段
        field = await reader.next()
        filename = field.filename
        
        # 保存文件
        size = 0
        with open(filename, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)
        
        return web.json_response({
            "filename": filename,
            "size": size
        })

# 5. 实时数据流
async def streaming_data():
    print("\n实时数据流示例：")
    
    async def stream_generator():
        for i in range(10):
            await asyncio.sleep(1)
            yield f"数据包 {i}\n"
    
    async def stream_handler(request):
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/plain'
        await response.prepare(request)
        
        async for data in stream_generator():
            await response.write(data.encode())
        
        return response

# 6. 代理服务器
class ProxyServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
    
    async def handle_client(self, reader, writer):
        # 读取客户端请求
        data = await reader.read(100)
        message = data.decode()
        addr = writer.get_extra_info('peername')
        
        print(f"收到来自 {addr} 的请求")
        
        # 转发到目标服务器
        try:
            target_reader, target_writer = await asyncio.open_connection(
                'target.example.com', 80)
            
            # 发送请求到目标
            target_writer.write(data)
            await target_writer.drain()
            
            # 获取响应并发送回客户端
            response = await target_reader.read(100)
            writer.write(response)
            await writer.drain()
            
            target_writer.close()
            await target_writer.wait_closed()
            
        except Exception as e:
            print(f"代理错误: {e}")
        
        writer.close()

    async def start(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port)
        
        addr = server.sockets[0].getsockname()
        print(f'代理服务器运行在 {addr}')
        
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    # 运行示例
    async def main():
        # 创建REST API服务器
        app = await rest_api_server()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        
        # 启动代理服务器
        proxy = ProxyServer()
        await proxy.start()
    
    asyncio.run(main()) 
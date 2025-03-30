# Python IO操作 - 网络IO示例

import socket
import select
import asyncio
import aiohttp
import requests
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

# 1. 基本Socket操作
def basic_socket_operations():
    print("基本Socket操作示例：")
    
    # TCP服务器
    def start_server():
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('localhost', 8888))
        server.listen(1)
        print("服务器启动在 localhost:8888")
        return server
    
    # TCP客户端
    def client_connect():
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 8888))
        return client
    
    # 启动服务器和客户端
    server = start_server()
    client = client_connect()
    
    # 接受连接
    conn, addr = server.accept()
    print(f"接受来自 {addr} 的连接")
    
    # 发送和接收数据
    client.send(b"Hello, Server!")
    data = conn.recv(1024)
    print(f"服务器收到: {data.decode()}")
    
    # 清理
    client.close()
    conn.close()
    server.close()

# 2. 非阻塞IO
def non_blocking_io():
    print("\n非阻塞IO示例：")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 8889))
    server.listen(5)
    server.setblocking(False)
    
    # 创建select监听列表
    inputs = [server]
    outputs = []
    
    try:
        while inputs:
            readable, writable, exceptional = select.select(inputs, outputs, inputs, 1)
            
            for s in readable:
                if s is server:
                    # 新连接
                    conn, addr = s.accept()
                    print(f"新连接从 {addr}")
                    conn.setblocking(False)
                    inputs.append(conn)
                else:
                    # 处理已存在的连接
                    try:
                        data = s.recv(1024)
                        if data:
                            print(f"收到数据: {data.decode()}")
                            if s not in outputs:
                                outputs.append(s)
                        else:
                            # 客户端断开连接
                            if s in outputs:
                                outputs.remove(s)
                            inputs.remove(s)
                            s.close()
                    except:
                        inputs.remove(s)
                        if s in outputs:
                            outputs.remove(s)
                        s.close()
    
    finally:
        server.close()

# 3. 异步IO
async def async_io():
    print("\n异步IO示例：")
    
    async def fetch_url(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()
    
    # 创建多个请求任务
    urls = [
        'http://example.com',
        'http://example.org',
        'http://example.net'
    ]
    
    tasks = [fetch_url(url) for url in urls]
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            print(f"获取 {url} 失败: {result}")
        else:
            print(f"获取 {url} 成功，内容长度: {len(result)}")

# 4. HTTP请求
def http_requests():
    print("\n HTTP请求示例：")
    
    # 使用urllib
    def urllib_example():
        with urlopen('http://example.com') as response:
            html = response.read()
            print(f"使用urllib获取内容长度: {len(html)}")
    
    # 使用requests
    def requests_example():
        response = requests.get('http://example.com')
        print(f"使用requests获取状态码: {response.status_code}")
        print(f"使用requests获取内容长度: {len(response.text)}")
    
    urllib_example()
    requests_example()

# 5. 并发网络请求
def concurrent_requests():
    print("\n并发网络请求示例：")
    
    def download_url(url):
        try:
            response = requests.get(url)
            return url, len(response.content)
        except Exception as e:
            return url, str(e)
    
    urls = [
        'http://example.com',
        'http://example.org',
        'http://example.net'
    ]
    
    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(download_url, urls)
        
        for url, result in zip(urls, results):
            if isinstance(result, int):
                print(f"下载 {url} 成功，大小: {result} 字节")
            else:
                print(f"下载 {url} 失败: {result}")

if __name__ == "__main__":
    basic_socket_operations()
    non_blocking_io()
    asyncio.run(async_io())
    http_requests()
    concurrent_requests() 
# Requests HTTP请求示例

import requests
import json

# 1. 基本GET请求
def basic_get_request():
    print("基本GET请求示例：")
    
    # 发送GET请求
    response = requests.get('https://api.github.com/events')
    print(f"状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")
    print(f"内容类型: {response.headers['content-type']}")
    
    # 打印部分响应内容
    if response.status_code == 200:
        data = response.json()[:2]  # 只显示前两条数据
        print(f"\n响应数据示例:\n{json.dumps(data, indent=2)}")

# 2. 带参数的请求
def requests_with_params():
    print("\n带参数的请求示例：")
    
    # GET请求带查询参数
    params = {
        'q': 'python',
        'sort': 'stars',
        'order': 'desc'
    }
    response = requests.get(
        'https://api.github.com/search/repositories',
        params=params
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"查询到的仓库数量: {data['total_count']}")
        # 显示第一个仓库的信息
        if data['items']:
            repo = data['items'][0]
            print(f"\n最受欢迎的Python仓库:")
            print(f"名称: {repo['name']}")
            print(f"星标数: {repo['stargazers_count']}")
            print(f"描述: {repo['description']}")

# 3. POST请求
def post_request():
    print("\n POST请求示例：")
    
    # 创建测试数据
    data = {
        'title': 'foo',
        'body': 'bar',
        'userId': 1
    }
    
    # 发送POST请求
    response = requests.post(
        'https://jsonplaceholder.typicode.com/posts',
        json=data
    )
    
    print(f"状态码: {response.status_code}")
    if response.status_code == 201:
        print(f"创建的帖子:\n{json.dumps(response.json(), indent=2)}")

# 4. 自定义请求头和错误处理
def custom_headers_and_errors():
    print("\n自定义请求头和错误处理示例：")
    
    # 自定义请求头
    headers = {
        'User-Agent': 'Python Requests Demo',
        'Accept': 'application/json'
    }
    
    try:
        # 发送请求
        response = requests.get(
            'https://api.github.com/user',
            headers=headers
        )
        
        # 检查状态码
        response.raise_for_status()
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"连接错误: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"超时错误: {timeout_err}")
    except requests.exceptions.RequestException as err:
        print(f"其他错误: {err}")
    else:
        print("请求成功！")
        if response.status_code == 200:
            print(f"响应数据:\n{json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    basic_get_request()
    requests_with_params()
    post_request()
    custom_headers_and_errors() 
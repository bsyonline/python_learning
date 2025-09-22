# MCP (Model Context Protocol) 示例集合

这是一个完整的MCP使用示例集合，展示了如何创建自定义MCP服务器并在LangGraph中使用它们。

## 🔧 环境设置

### 1. 安装依赖
```bash
pip install mcp langchain-mcp-adapters langgraph langchain-openai python-dotenv psutil
```

### 2. 设置API密钥
创建 `.env` 文件并添加：
```
API_KEY=your_api_key_here
```

## 📁 文件说明

### MCP服务器
- `math_tool_mcp_server.py` - 数学计算工具服务器
- `weather_tool_mcp_server.py` - 天气查询服务器 (HTTP传输)
- `host_info_tool_mcp_server.py` - 系统信息查询服务器
- `file_management_mcp_server.py` - 文件管理工具服务器 (新增)
- `data_processing_mcp_server.py` - 数据处理工具服务器 (新增)

### 客户端示例
- `simple_mcp_demo.py` - 🌟 快速上手示例 (推荐新手)
- `complete_mcp_example.py` - 完整功能演示
- `mcp_client.py` - 原始客户端示例
- `start_servers.py` - 服务器启动脚本

## 🚀 快速开始

### 方法1：简单示例 (推荐)
```bash
# 直接运行简单示例
python simple_mcp_demo.py
```

### 方法2：完整演示
```bash
# 运行完整示例 (包含交互模式)
python complete_mcp_example.py
```

### 方法3：分别启动服务器
```bash
# 终端1：启动天气服务器 (HTTP)
python weather_tool_mcp_server.py

# 终端2：启动客户端
python mcp_client.py
```

### 方法4：同时启动多个服务器
```bash
# 终端1：启动所有HTTP服务器
python start_servers.py

# 终端2：启动客户端
python mcp_client.py
```

## 🛠️ MCP服务器详解

### 1. 数学工具服务器
提供基础数学计算功能：
- 四则运算 (add, subtract, multiply, divide)
- 高级数学 (power, square_root, factorial)
- 几何计算 (calculate_area)

### 2. 文件管理服务器 (新增)
提供文件系统操作：
- 目录列表 (list_directory)
- 文件读取 (read_file_content)
- 文件信息 (get_file_info)
- 文件搜索 (search_files)
- 创建目录 (create_directory)
- 写入文件 (write_file)

### 3. 数据处理服务器 (新增)
提供数据分析功能：
- 数字统计分析 (analyze_numbers)
- CSV数据处理 (process_csv_data)
- 数据过滤排序 (filter_data, sort_data)
- 相关性分析 (calculate_correlation)
- 报告生成 (generate_report)

### 4. 天气服务器
模拟天气查询API：
- 当前天气 (get_current_weather)
- 天气预报 (get_weather_forecast)
- 城市对比 (compare_weather)

### 5. 系统信息服务器
获取主机信息：
- 系统配置
- 硬件信息
- 内存和CPU信息

## 🎯 使用场景示例

### 数学计算
```
用户: "计算 (15 + 7) × 3 的结果，并求平方根"
AI: 使用数学工具计算并返回结果
```

### 文件操作
```
用户: "列出当前目录的所有Python文件"
AI: 使用文件管理工具搜索.py文件
```

### 数据分析
```
用户: "分析这些数字：[10, 20, 30, 40, 50]"
AI: 使用数据处理工具计算统计信息
```

### 系统查询
```
用户: "查看我的电脑配置"
AI: 使用系统信息工具获取硬件信息
```

## 🔧 传输协议说明

- **stdio传输**: 通过标准输入输出与MCP服务器通信，适合本地工具
- **HTTP传输**: 通过HTTP接口与MCP服务器通信，适合网络服务

## 🚨 故障排除

1. **权限错误**: 确保有文件读写权限
2. **端口占用**: 天气服务器默认使用8000端口
3. **依赖缺失**: 检查是否安装了所有必要的包
4. **API密钥**: 确保.env文件中设置了正确的API_KEY

## 📚 扩展指南

要创建自己的MCP服务器：

1. 导入FastMCP
2. 使用@mcp.tool()装饰器定义工具函数
3. 选择合适的传输协议
4. 在客户端配置中添加新服务器

示例：
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def my_function(param: str) -> str:
    """我的自定义函数"""
    return f"处理: {param}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```


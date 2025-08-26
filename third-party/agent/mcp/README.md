# 本地MCP服务注册指南

## 概述

这是一个主机信息MCP服务，提供系统硬件信息查询功能。

## 文件结构

- `mcp_server.py` - MCP服务器主文件
- `mcp_tools.py` - MCP工具函数定义
- `.trae/mcp.json` - Trae MCP配置文件

## 功能

提供以下主机信息：
- 操作系统信息
- 处理器信息
- 内存信息
- CPU核心数
- CPU型号（如果可用）

## 在Trae中注册本地MCP服务

### 方法一：自动配置（推荐）

Trae会自动读取项目根目录下的 `.trae/mcp.json` 配置文件。我们已经为您创建了这个文件，配置内容如下：

```json
{
  "mcpServers": {
    "host-info-mcp": {
      "command": "python",
      "args": [
        "d:\\Dev\\workspace\\python\\python_learning\\third-party\\agent\\mcp\\mcp_server.py"
      ],
      "env": {},
      "cwd": "d:\\Dev\\workspace\\python\\python_learning"
    }
  }
}
```

### 方法二：手动配置

如果自动配置不生效，您可以手动在Trae中配置：

1. 打开Trae IDE
2. 进入设置 → MCP设置
3. 添加新的MCP服务器
4. 配置如下：
   - 名称：host-info-mcp
   - 命令：python
   - 参数：`d:\Dev\workspace\python\python_learning\third-party\agent\mcp\mcp_server.py`
   - 工作目录：`d:\Dev\workspace\python\python_learning`

## 使用方法

在Trae中注册成功后，您可以通过以下方式使用：

1. 在聊天界面中询问AI助手关于主机信息
2. AI会自动调用MCP服务获取系统信息
3. 返回格式化的JSON数据

## 测试服务

您可以手动运行服务进行测试：

```bash
cd d:\Dev\workspace\python\python_learning\third-party\agent\mcp
python mcp_server.py
```

或者测试工具函数：

```bash
python mcp_tools.py
```

## 依赖要求

确保已安装以下Python包：

```bash
pip install mcp-server-fastmcp psutil
```

## 故障排除

如果服务无法正常启动：

1. 检查Python路径是否正确
2. 确认依赖包已安装
3. 查看Trae的MCP服务器日志
4. 确保文件路径中的反斜杠正确转义

## 扩展功能

您可以修改 `mcp_tools.py` 文件来添加更多工具函数，然后在 `mcp_server.py` 中注册使用。
import subprocess
import sys
import time
import os

def start_servers():
    print("🚀 Starting MCP Servers...")
    
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    math_server_path = os.path.join(current_dir, "math_tool_mcp_server.py")
    weather_server_path = os.path.join(current_dir, "weather_tool_mcp_server.py")
    host_info_server_path = os.path.join(current_dir, "host_info_tool_mcp_server.py")
    
    processes = []
    
    try:
        # 启动天气服务器 (HTTP)
        print("🌤️  Starting Weather Server on port 8000...")
        weather_process = subprocess.Popen(
            [sys.executable, weather_server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(weather_process)
        
        # 等待一下让天气服务器启动
        time.sleep(2)
        
        # 启动数学服务器 (stdio)
        print("🧮 Starting Math Server (stdio transport)...")
        math_process = subprocess.Popen(
            [sys.executable, math_server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(math_process)

        # 启动主机信息服务器 (stdio)
        print("🖥️  Starting Host Info Server (stdio transport)...")
        host_info_process = subprocess.Popen(
            [sys.executable, host_info_server_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(host_info_process)
        
        print("✅ Servers started successfully!")
        print("\n📋 Server Information:")
        print("   Math Server: stdio transport (ready for MCP client)")
        print("   Weather Server: http://localhost:8000/mcp")
        print("   Host Info Server: stdio transport (ready for MCP client)")
        print("\n💡 Now you can run: python mcp_client.py")
        print("\n⏹️  Press Ctrl+C to stop all servers")
        
        # 等待用户中断
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        for process in processes:
            process.terminate()
        print("✅ Servers stopped")
    
    except Exception as e:
        print(f"❌ Error starting servers: {e}")
        for process in processes:
            process.terminate()

if __name__ == "__main__":
    start_servers()
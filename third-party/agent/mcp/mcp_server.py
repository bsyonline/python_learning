from mcp.server.fastmcp import FastMCP
import mcp_tools

mcp = FastMCP("host info mcp")
mcp.add_tool(mcp_tools.get_host_info) # 显示注册tool，也可以通过装饰器@mcp.tool()自动注册

def main():
    mcp.run("stdio") # sse


if __name__ == "__main__":
    main()

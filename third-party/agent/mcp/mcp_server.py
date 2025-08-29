from fastmcp import FastMCP
import mcp_tools

mcp = FastMCP("host-info-mcp")

@mcp.tool()
def get_host_info():
    return mcp_tools.get_host_info()

def main():
    mcp.run("stdio") 

if __name__ == "__main__":
    main()
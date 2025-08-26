from fastmcp import FastMCP
import mcp_tools

mcp = FastMCP("host info mcp")

def main():
    mcp.run("stdio") # sse


if __name__ == "__main__":
    main()

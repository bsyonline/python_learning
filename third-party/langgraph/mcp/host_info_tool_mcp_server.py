from mcp.server.fastmcp import FastMCP
import platform
import psutil
import subprocess
import json

mcp = FastMCP("HostInfoTools")

@mcp.tool()
def get_host_info() -> str:
    """get host information
    Returns:
        str: the host information in JSON string
    """
    info: dict[str, str] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "memory_gb": str(round(psutil.virtual_memory().total / (1024**3), 2)),
    }

    cpu_count = psutil.cpu_count(logical=True)
    info["cpu_count"] = str(cpu_count) if cpu_count is not None else "-1"
    
    try:
        cpu_model = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        ).decode().strip()
        info["cpu_model"] = cpu_model
    except Exception:
        info["cpu_model"] = "Unknown"

    return json.dumps(info, indent=4)

if __name__ == "__main__":
    print("Starting Math MCP server on stdio transport...")
    mcp.run(transport="stdio")
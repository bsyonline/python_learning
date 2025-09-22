from mcp.server.fastmcp import FastMCP
import math

mcp = FastMCP("MathTools")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers"""
    print(f"-----> mcp math tool add {a} {b}")
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    print(f"-----> mcp math tool subtract {a} {b}")
    return a - b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    print(f"-----> mcp math tool multiply {a} {b}")
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b"""
    print(f"-----> mcp math tool divide {a} {b}")
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent"""
    print(f"-----> mcp math tool power {base} {exponent}")
    return base ** exponent

@mcp.tool()
def square_root(x: float) -> float:
    """Calculate square root of a number"""
    print(f"-----> mcp math tool square_root {x}")
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(x)

@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of a number"""
    print(f"-----> mcp math tool factorial {n}")
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)

@mcp.tool()
def calculate_area(shape: str, **kwargs) -> float:
    """Calculate area of different shapes. 
    
    Supported shapes:
    - 'circle': requires 'radius'
    - 'rectangle': requires 'width' and 'height'
    - 'triangle': requires 'base' and 'height'
    """
    print(f"-----> mcp math tool calculate_area {shape} {kwargs}")  
    shape = shape.lower()
    
    if shape == 'circle':
        radius = kwargs.get('radius')
        if radius is None:
            raise ValueError("Circle requires radius parameter")
        return math.pi * radius ** 2
    
    elif shape == 'rectangle':
        width = kwargs.get('width')
        height = kwargs.get('height')
        if width is None or height is None:
            raise ValueError("Rectangle requires width and height parameters")
        return width * height
    
    elif shape == 'triangle':
        base = kwargs.get('base')
        height = kwargs.get('height')
        if base is None or height is None:
            raise ValueError("Triangle requires base and height parameters")
        return 0.5 * base * height
    
    else:
        raise ValueError(f"Unsupported shape: {shape}")

if __name__ == "__main__":
    print("Starting Math MCP server on stdio transport...")
    mcp.run(transport="stdio")
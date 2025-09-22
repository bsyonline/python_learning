from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta
import random

mcp = FastMCP("WeatherService")

# 模拟一些城市的天气数据
CITY_WEATHER = {
    "beijing": {
        "current": {"temp": 25, "condition": "sunny", "humidity": 45},
        "forecast": [
            {"day": "today", "high": 28, "low": 18, "condition": "sunny"},
            {"day": "tomorrow", "high": 26, "low": 17, "condition": "cloudy"},
            {"day": "day after", "high": 24, "low": 16, "condition": "rainy"}
        ]
    },
    "shanghai": {
        "current": {"temp": 22, "condition": "cloudy", "humidity": 65},
        "forecast": [
            {"day": "today", "high": 24, "low": 19, "condition": "cloudy"},
            {"day": "tomorrow", "high": 23, "low": 18, "condition": "rainy"},
            {"day": "day after", "high": 25, "low": 20, "condition": "sunny"}
        ]
    },
    "new york": {
        "current": {"temp": 18, "condition": "rainy", "humidity": 75},
        "forecast": [
            {"day": "today", "high": 20, "low": 15, "condition": "rainy"},
            {"day": "tomorrow", "high": 22, "low": 16, "condition": "cloudy"},
            {"day": "day after", "high": 24, "low": 18, "condition": "sunny"}
        ]
    },
    "london": {
        "current": {"temp": 12, "condition": "foggy", "humidity": 80},
        "forecast": [
            {"day": "today", "high": 14, "low": 8, "condition": "foggy"},
            {"day": "tomorrow", "high": 15, "low": 9, "condition": "cloudy"},
            {"day": "day after", "high": 16, "low": 10, "condition": "rainy"}
        ]
    }
}

@mcp.tool()
async def get_current_weather(location: str) -> dict:
    """Get current weather for a specific location"""
    print(f"-----> mcp weather tool get_current_weather {location}")
    location_lower = location.lower()
    
    if location_lower not in CITY_WEATHER:
        # 对于未知城市，生成随机天气数据
        return {
            "location": location,
            "temperature": random.randint(5, 35),
            "condition": random.choice(["sunny", "cloudy", "rainy", "snowy"]),
            "humidity": random.randint(30, 90),
            "source": "simulated"
        }
    
    data = CITY_WEATHER[location_lower]["current"]
    return {
        "location": location,
        "temperature": data["temp"],
        "condition": data["condition"],
        "humidity": data["humidity"],
        "source": "predefined"
    }

@mcp.tool()
async def get_weather_forecast(location: str, days: int = 3) -> dict:
    """Get weather forecast for a location for specified number of days"""
    print(f"-----> mcp weather tool get_weather_forecast {location} {days}")
    location_lower = location.lower()
    
    if location_lower not in CITY_WEATHER:
        # 为未知城市生成预测
        forecast = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            forecast.append({
                "date": date,
                "high": random.randint(10, 30),
                "low": random.randint(0, 20),
                "condition": random.choice(["sunny", "cloudy", "rainy", "snowy"])
            })
        return {
            "location": location,
            "forecast": forecast,
            "source": "simulated"
        }
    
    data = CITY_WEATHER[location_lower]["forecast"][:days]
    return {
        "location": location,
        "forecast": data,
        "source": "predefined"
    }

@mcp.tool()
async def compare_weather(location1: str, location2: str) -> dict:
    """Compare weather between two locations"""
    print(f"-----> mcp weather tool compare_weather {location1} {location2}")
    weather1 = await get_current_weather(location1)
    weather2 = await get_current_weather(location2)
    
    return {
        "comparison": {
            location1: weather1,
            location2: weather2,
            "temperature_difference": abs(weather1["temperature"] - weather2["temperature"])
        }
    }

if __name__ == "__main__":
    print("Starting Weather MCP server on HTTP transport (port 8000)...")
    mcp.run(transport="streamable-http")
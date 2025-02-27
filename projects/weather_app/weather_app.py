# 天气查询应用

import requests
import json
from datetime import datetime

class WeatherApp:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city):
        """获取城市天气信息"""
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",  # 使用摄氏度
            "lang": "zh_cn"     # 使用中文
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取天气信息失败: {e}")
            return None
    
    def format_weather_data(self, weather_data):
        """格式化天气数据"""
        if not weather_data:
            return "无法获取天气信息"
        
        try:
            city = weather_data["name"]
            temp = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            description = weather_data["weather"][0]["description"]
            
            return f"""
城市: {city}
温度: {temp}°C
湿度: {humidity}%
风速: {wind_speed}m/s
天气: {description}
更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        except KeyError:
            return "天气数据格式错误"
    
    def save_weather_history(self, city, weather_data):
        """保存天气查询历史"""
        if not weather_data:
            return
        
        history = self.load_weather_history()
        history.append({
            "city": city,
            "data": weather_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        with open("weather_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
    
    def load_weather_history(self):
        """加载天气查询历史"""
        try:
            with open("weather_history.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

def main():
    # 请替换为你的 OpenWeatherMap API 密钥
    API_KEY = "your_api_key_here"
    app = WeatherApp(API_KEY)
    
    while True:
        print("\n天气查询应用")
        print("1. 查询天气")
        print("2. 查看历史记录")
        print("3. 退出")
        
        choice = input("请选择操作 (1-3): ")
        
        if choice == "1":
            city = input("请输入城市名称: ")
            weather_data = app.get_weather(city)
            
            if weather_data:
                print("\n天气信息:")
                print(app.format_weather_data(weather_data))
                app.save_weather_history(city, weather_data)
        
        elif choice == "2":
            history = app.load_weather_history()
            if history:
                print("\n查询历史:")
                for record in history:
                    print(f"\n时间: {record['timestamp']}")
                    print(app.format_weather_data(record['data']))
            else:
                print("暂无查询历史")
        
        elif choice == "3":
            print("感谢使用！")
            break
        
        else:
            print("无效的选择")

if __name__ == "__main__":
    main() 
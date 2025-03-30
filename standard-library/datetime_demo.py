# Python标准库 - 日期和时间处理

from datetime import datetime, date, time, timedelta
import time as time_lib

# 1. 基本日期时间操作
def basic_datetime():
    print("基本日期时间操作示例：")
    
    # 获取当前日期和时间
    now = datetime.now()
    print(f"当前日期时间: {now}")
    
    # 创建特定日期时间
    specific_date = datetime(2024, 1, 1, 12, 30, 0)
    print(f"特定日期时间: {specific_date}")
    
    # 格式化日期时间
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"格式化后: {formatted}")

# 2. 日期计算
def date_calculations():
    print("\n日期计算示例：")
    
    today = date.today()
    print(f"今天: {today}")
    
    # 日期加减
    tomorrow = today + timedelta(days=1)
    print(f"明天: {tomorrow}")
    
    next_week = today + timedelta(weeks=1)
    print(f"下周: {next_week}")
    
    # 计算日期差
    date1 = date(2024, 1, 1)
    date2 = date(2024, 12, 31)
    diff = date2 - date1
    print(f"2024年天数: {diff.days}")

# 3. 时间戳操作
def timestamp_operations():
    print("\n时间戳操作示例：")
    
    # 获取当前时间戳
    timestamp = time_lib.time()
    print(f"当前时间戳: {timestamp}")
    
    # 时间戳转datetime
    dt = datetime.fromtimestamp(timestamp)
    print(f"时间戳转换为日期时间: {dt}")
    
    # datetime转时间戳
    timestamp_back = dt.timestamp()
    print(f"日期时间转换为时间戳: {timestamp_back}")

# 4. 时区处理
def timezone_handling():
    print("\n时区处理示例：")
    from datetime import timezone, timedelta
    
    # UTC时间
    utc_now = datetime.now(timezone.utc)
    print(f"UTC时间: {utc_now}")
    
    # 创建特定时区
    beijing_tz = timezone(timedelta(hours=8))
    beijing_time = datetime.now(beijing_tz)
    print(f"北京时间: {beijing_time}")

if __name__ == "__main__":
    basic_datetime()
    date_calculations()
    timestamp_operations()
    timezone_handling() 
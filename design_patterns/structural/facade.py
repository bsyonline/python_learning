# Python设计模式 - 外观模式
# 为子系统中的一组接口提供一个一致的界面，使得子系统更容易使用

from typing import List, Dict
import time
import json

# 1. 复杂子系统类
class CPU:
    """CPU子系统"""
    
    def freeze(self) -> None:
        print("CPU: 冻结所有进程")
    
    def jump(self, address: str) -> None:
        print(f"CPU: 跳转到地址 {address}")
    
    def execute(self) -> None:
        print("CPU: 执行指令")

class Memory:
    """内存子系统"""
    
    def load(self, address: str, data: str) -> None:
        print(f"内存: 加载数据 '{data}' 到地址 {address}")

class HardDrive:
    """硬盘子系统"""
    
    def read(self, sector: str, size: int) -> str:
        print(f"硬盘: 从扇区 {sector} 读取 {size} 字节")
        return "数据"

# 2. 外观类
class ComputerFacade:
    """计算机外观类"""
    
    def __init__(self):
        self._cpu = CPU()
        self._memory = Memory()
        self._hard_drive = HardDrive()
    
    def start(self) -> None:
        """启动计算机"""
        print("\n开始启动计算机...")
        self._cpu.freeze()
        self._memory.load("BOOT_ADDRESS", "BOOT_DATA")
        self._cpu.jump("BOOT_ADDRESS")
        self._cpu.execute()

# 3. 复杂子系统 - 多媒体处理
class VideoProcessor:
    """视频处理器"""
    
    def process(self, video: str) -> str:
        print(f"处理视频: {video}")
        return f"{video}_processed"

class AudioProcessor:
    """音频处理器"""
    
    def process(self, audio: str) -> str:
        print(f"处理音频: {audio}")
        return f"{audio}_processed"

class ImageProcessor:
    """图像处理器"""
    
    def process(self, image: str) -> str:
        print(f"处理图像: {image}")
        return f"{image}_processed"

# 4. 多媒体处理外观
class MediaFacade:
    """多媒体处理外观"""
    
    def __init__(self):
        self._video = VideoProcessor()
        self._audio = AudioProcessor()
        self._image = ImageProcessor()
    
    def process_media(self, media_files: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """处理多媒体文件"""
        result = {}
        
        # 处理视频
        if "videos" in media_files:
            result["videos"] = [
                self._video.process(video)
                for video in media_files["videos"]
            ]
        
        # 处理音频
        if "audios" in media_files:
            result["audios"] = [
                self._audio.process(audio)
                for audio in media_files["audios"]
            ]
        
        # 处理图像
        if "images" in media_files:
            result["images"] = [
                self._image.process(image)
                for image in media_files["images"]
            ]
        
        return result

# 5. 复杂子系统 - 数据库操作
class DatabaseConnection:
    """数据库连接"""
    
    def connect(self) -> None:
        print("连接到数据库")
    
    def disconnect(self) -> None:
        print("断开数据库连接")

class QueryBuilder:
    """查询构建器"""
    
    def build_query(self, table: str, conditions: Dict) -> str:
        return f"SELECT * FROM {table} WHERE {json.dumps(conditions)}"

class ResultFormatter:
    """结果格式化器"""
    
    def format(self, data: List) -> str:
        return json.dumps(data, indent=2)

# 6. 数据库操作外观
class DatabaseFacade:
    """数据库操作外观"""
    
    def __init__(self):
        self._connection = DatabaseConnection()
        self._query_builder = QueryBuilder()
        self._formatter = ResultFormatter()
    
    def get_formatted_data(self, table: str, conditions: Dict) -> str:
        """获取格式化的数据"""
        print("\n执行数据库操作...")
        self._connection.connect()
        
        query = self._query_builder.build_query(table, conditions)
        print(f"执行查询: {query}")
        
        # 模拟查询结果
        result = [{"id": 1, "name": "示例数据"}]
        formatted_result = self._formatter.format(result)
        
        self._connection.disconnect()
        return formatted_result

# 7. 使用示例
def facade_demo():
    print("外观模式示例：")
    
    # 计算机启动示例
    print("\n1. 计算机启动示例:")
    computer = ComputerFacade()
    computer.start()
    
    # 多媒体处理示例
    print("\n2. 多媒体处理示例:")
    media_facade = MediaFacade()
    media_files = {
        "videos": ["video1.mp4", "video2.mp4"],
        "audios": ["audio1.mp3"],
        "images": ["image1.jpg", "image2.png"]
    }
    
    processed_files = media_facade.process_media(media_files)
    print("\n处理结果:")
    print(json.dumps(processed_files, indent=2))
    
    # 数据库操作示例
    print("\n3. 数据库操作示例:")
    db_facade = DatabaseFacade()
    result = db_facade.get_formatted_data(
        "users",
        {"age": {"$gt": 18}}
    )
    print(f"\n查询结果:\n{result}")

if __name__ == "__main__":
    facade_demo() 
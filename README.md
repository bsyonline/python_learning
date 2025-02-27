# Python学习项目

这是一个用于学习Python的结构化项目。该项目包含多个模块，帮助你系统地学习Python编程。

## 项目结构

- `basics/`: Python基础知识
- `data_structures/`: Python数据结构
- `standard_library/`: Python标准库示例
  - 日期时间处理
  - JSON处理
  - 操作系统接口
  - 随机数生成
  - 线程操作
  - 进程操作
- `third_party/`: 第三方库学习
- `projects/`: 实践项目
- `tests/`: 单元测试

## 如何使用

1. 克隆此仓库
2. 安装依赖: `pip install -r requirements.txt`
3. 按照目录结构逐步学习
4. 每个模块都包含示例代码和注释说明

## 学习顺序建议

1. 从 `basics/` 开始，学习Python基础知识
2. 继续学习 `data_structures/` 中的数据结构
3. 探索 `standard_library/` 中的标准库用法
    - 包括线程和进程的并发编程
4. 学习 `third_party/` 中的常用第三方库
5. 在 `projects/` 中实践所学知识

## 依赖管理

项目使用的第三方库都列在 `requirements.txt` 中。

## 并发编程示例

项目包含了完整的并发编程示例：

### 线程编程 (threading_demo.py)
- 基本线程创建和使用
- 线程同步（使用Lock）
- 线程间通信（使用Queue）
- 线程池的使用

### 进程编程 (multiprocessing_demo.py)
- 基本进程创建和使用
- 进程间通信（使用Queue）
- 进程池操作
- 共享内存示例 
# Hitchcock Dolly Zoom Generator

这是一个用于生成**希区柯克变焦 (Dolly Zoom)** 效果的工具。它能够自动追踪视频中的主体，并应用反向缩放效果，创造出背景压缩或延伸的视觉冲击力。

## ✨ 特性

- **智能追踪**：结合 OpenCV GrabCut 和 CSRT 算法，自动锁定并追踪主体。
- **强力防抖**：内置 Kalman 滤波和滑动窗口平滑算法，消除画面抖动。
- **双模式运行**：
  - 🖥️ **Web 界面**：基于 Gradio 的可视化操作，支持涂抹选区。
  - ⚙️ **命令行**：适合批处理或脚本调用。
- **浏览器友好**：自动转码为 H.264 (MP4)，确保在网页端流畅播放。

## 🛠️ 环境准备

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. FFmpeg 环境
本项目内置了 `imageio-ffmpeg`，通常无需手动安装系统级 FFmpeg。
但如果遇到解码问题，建议安装系统 FFmpeg：

- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`
- **Windows**: 下载配置环境变量

## 🚀 运行方法

### 方式一：Web 界面 (推荐)

启动可视化界面：

```bash
python app.py
```

然后浏览器访问终端显示的地址 (通常是 http://127.0.0.1:7860)。

**操作步骤：**

1. 上传视频。
2. 点击 **"提取第一帧"**。
3. 在图片上**涂抹**你要追踪的主体（大概涂满主体即可）。
4. 点击 **"生成效果"**。

### 方式二：命令行工具

直接处理视频文件：

```bash
python main.py -i input/your_video.mp4 -o output/result.mp4
```

程序会弹出一个窗口，请**框选**主体并按 **空格键** 确认。

## 📂 项目结构

- `app.py`: Gradio Web 应用入口。
- `main.py`: 命令行工具入口。
- `core.py`: 核心算法实现 (DollyZoomProcessor)。
- `requirements.txt`: Python 依赖列表。

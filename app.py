import gradio as gr
import cv2
import numpy as np
import os
from core import DollyZoomProcessor
import socket

def get_first_frame(video_path):
    if not video_path:
        return None
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def process_video(video_path, sketch_data, progress=gr.Progress()):
    if not video_path:
        return None
    if sketch_data is None:
        return None

    # 获取视频尺寸，确保 Mask 与视频尺寸一致
    cap = cv2.VideoCapture(video_path)
    ret, v_frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("无法读取视频")
    
    video_h, video_w = v_frame.shape[:2]
        
    # 解析 Gradio 的 sketch 数据 (Gradio 3.x)
    mask = None
    if isinstance(sketch_data, dict):
        mask = sketch_data.get('mask')
        if mask is None:
            # 兼容其他可能的结构
            mask = sketch_data.get('layers', [None])[0] or sketch_data.get('composite')
    
    if mask is None:
        raise ValueError("无法获取涂抹遮罩 (Mask)，请确保在图片上进行了涂抹")

    # 调整 Mask 尺寸以适应视频帧
    if mask.shape[0] != video_h or mask.shape[1] != video_w:
        print(f"Resizing mask from {mask.shape[1]}x{mask.shape[0]} to {video_w}x{video_h}")
        mask = cv2.resize(mask, (video_w, video_h), interpolation=cv2.INTER_NEAREST)

    # 处理 Mask (RGBA -> Binary)
    if mask.ndim == 3 and mask.shape[2] == 4:
        # 提取 Alpha 通道
        # 涂抹的地方 Alpha > 0
        _, _, _, alpha = cv2.split(mask)
        mask_gray = alpha
    elif mask.ndim == 3:
        # RGB 转灰度
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask

    # 二值化确保只有涂抹区域
    _, mask_binary = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)

    # 寻找 Bounding Box
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 如果找不到轮廓（可能是整个图都被涂了，或者阈值问题）
        if np.count_nonzero(mask_binary) > 0:
            y_idxs, x_idxs = np.nonzero(mask_binary)
            y_min, y_max = np.min(y_idxs), np.max(y_idxs)
            x_min, x_max = np.min(x_idxs), np.max(x_idxs)
            initial_bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
        else:
            raise ValueError("未检测到涂抹区域，请在主体上进行涂抹")
    else:
        # 合并所有轮廓
        all_cnts = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_cnts)
        initial_bbox = (x, y, w, h)
    
    print(f"User selected bbox: {initial_bbox}")
    
    # 输出路径
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "gradio_output.mp4")
    
    # 进度回调
    def update_progress(curr, total):
        progress(curr / total, desc=f"Processing Frame {curr}/{total}")
        
    processor = DollyZoomProcessor(video_path, output_path)
    processor.process(initial_bbox, progress_callback=update_progress)
    
    return output_path

with gr.Blocks(title="Hitchcock Dolly Zoom Effect") as demo:
    gr.Markdown("# Hitchcock Dolly Zoom Generator")
    gr.Markdown("上传视频 -> 提取第一帧 -> 涂抹主体 -> 生成大片！")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="1. 上传视频素材")
            extract_btn = gr.Button("提取第一帧", variant="primary")
            
        with gr.Column():
            # Gradio 3.x 使用 Image(tool="sketch")
            image_input = gr.Image(
                label="2. 涂抹要追踪的主体", 
                type="numpy", 
                tool="sketch",
                interactive=True
            )
            run_btn = gr.Button("生成效果", variant="primary")
            
    video_output = gr.Video(label="3. 生成结果")
    
    # 事件绑定
    extract_btn.click(
        fn=get_first_frame,
        inputs=[video_input],
        outputs=[image_input]
    )
    
    run_btn.click(
        fn=process_video,
        inputs=[video_input, image_input],
        outputs=[video_output]
    )

if __name__ == "__main__":
    # 设定端口
    port = 7860
    
    # --- 新增：获取并打印局域网 IP 的逻辑 ---
    try:
        # 创建一个 UDP 套接字
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 尝试连接一个公共 DNS (不会真的发送数据，只是为了让系统判断出路由出口 IP)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        # 使用 ANSI 转义码打印绿色高亮文本，方便在终端一眼看到
        print("-" * 40)
        print(f"✅  本地访问链接 (本机使用):")
        print(f"\033[92m   http://127.0.0.1:{port} \033[0m")
        print(f"\n✅  局域网访问链接 (复制给同一WiFi下的同学):")
        print(f"\033[92m   http://{local_ip}:{port} \033[0m") # 绿色字体
        print("-" * 40)
    except Exception as e:
        print(f"⚠️ 无法自动获取局域网 IP: {e}")
    # ---------------------------------------

    # 启动 Gradio
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=port,
        share=False
    )
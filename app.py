import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from core import DollyZoomProcessor
import socket
import imageio_ffmpeg
import subprocess

def convert_to_h264(video_path, progress=gr.Progress()):
    """
    将视频转换为浏览器兼容的 H.264 MP4 格式
    """
    if not video_path:
        return None
        
    progress(0, desc="正在准备转码...")
    
    # 创建临时输出文件
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # 构建 ffmpeg 命令
    # -y: 覆盖输出文件
    # -c:v libx264: 使用 H.264 视频编码
    # -c:a aac: 使用 AAC 音频编码
    # -preset fast: 编码速度优先
    # -crf 23: 默认质量
    # -loglevel error: 屏蔽日志
    cmd = [
        ffmpeg_exe, '-y',
        '-i', video_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'fast',
        '-loglevel', 'error',
        output_path
    ]
    
    progress(0.1, desc="正在转码 (这可能需要一点时间)...")
    
    try:
        # 运行转码
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"转码失败: {e}")
        # 如果转码失败，返回原视频（虽然可能还是播不了，但至少不报错）
        return video_path

def get_first_frame(video_path, reverse=False):
    if not video_path:
        return None
    
    cap = cv2.VideoCapture(video_path)
    if reverse:
        # 如果倒放，读取最后一帧作为“第一帧”供用户涂抹
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def reverse_video_file(video_path):
    """倒放视频并返回临时文件路径"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 使用 tempfile 创建临时文件
    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    # out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    # 使用 FFmpeg 管道写入 H.264
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',
        '-loglevel', 'error', # 屏蔽日志
        temp_path
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # 内存警告：如果视频过长，这里会消耗大量内存
    # 实际生产环境建议分块处理或限制时长
    
    for frame in reversed(frames):
        # out.write(frame)
        try:
            p.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"Error writing to ffmpeg reverse: {e}")
            break
            
    # out.release()
    if p.stdin:
        p.stdin.close()
    p.wait()
    
    return temp_path

def process_video(video_path, sketch_data, reverse_input=False, reverse_output=False, progress=gr.Progress()):
    if not video_path:
        return None
    if sketch_data is None:
        return None

    # 处理倒放逻辑
    target_video_path = video_path
    if reverse_input:
        progress(0, desc="Reversing input video...")
        try:
            target_video_path = reverse_video_file(video_path)
        except Exception as e:
            raise ValueError(f"倒放视频失败 (可能是视频过长导致内存不足): {e}")

    # 获取视频尺寸，确保 Mask 与视频尺寸一致
    cap = cv2.VideoCapture(target_video_path)
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
    
    # 临时输出路径（如果是倒放输出，这只是中间文件）
    temp_output_filename = "temp_dolly_output.mp4" if reverse_output else "gradio_output.mp4"
    output_path = os.path.join(output_dir, temp_output_filename)
    
    # 进度回调
    def update_progress(curr, total):
        progress(curr / total, desc=f"Processing Frame {curr}/{total}")
        
    processor = DollyZoomProcessor(target_video_path, output_path)
    processor.process(initial_bbox, progress_callback=update_progress)
    
    # 清理输入临时文件
    if reverse_input and target_video_path != video_path and os.path.exists(target_video_path):
        try:
            os.remove(target_video_path)
        except:
            pass
            
    # 处理输出倒放逻辑
    if reverse_output:
        progress(0, desc="Reversing output video...")
        final_output_path = os.path.join(output_dir, "gradio_output_reversed.mp4")
        try:
            reversed_temp_path = reverse_video_file(output_path)
            # 移动/重命名临时文件到最终输出路径
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            os.rename(reversed_temp_path, final_output_path)
            
            # 删除中间生成的正向 Dolly Zoom 视频
            if os.path.exists(output_path):
                os.remove(output_path)
                
            return final_output_path
        except Exception as e:
            print(f"Error reversing output: {e}")
            # 如果倒放失败，返回正向的
            return output_path
            
    return output_path

with gr.Blocks(title="Hitchcock Dolly Zoom Effect") as demo:
    gr.Markdown("# Hitchcock Dolly Zoom Generator")
    gr.Markdown("上传视频 -> 提取第一帧 -> 涂抹主体 -> 生成大片！")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="1. 上传视频素材", interactive=True, height=360)
            convert_btn = gr.Button("⚠️ 视频无法播放？点击转码", variant="secondary")
            with gr.Row():
                reverse_input_check = gr.Checkbox(label="倒放输入视频", value=False)
                reverse_output_check = gr.Checkbox(label="倒放生成结果", value=False)
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
    convert_btn.click(
        fn=convert_to_h264,
        inputs=[video_input],
        outputs=[video_input]
    )

    extract_btn.click(
        fn=get_first_frame,
        inputs=[video_input, reverse_input_check],
        outputs=[image_input]
    )
    
    run_btn.click(
        fn=process_video,
        inputs=[video_input, image_input, reverse_input_check, reverse_output_check],
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
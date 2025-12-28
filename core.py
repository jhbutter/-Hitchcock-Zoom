import cv2
import numpy as np
import os
import imageio_ffmpeg
import subprocess

class DollyZoomProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 获取 ffmpeg 路径
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
    def _get_ffmpeg_writer(self, output_path, fps, width, height):
        """创建一个 FFmpeg 管道写入器"""
        cmd = [
            self.ffmpeg_exe,
            '-y', # 覆盖输出
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24', # OpenCV 输出是 BGR
            '-r', str(fps),
            '-i', '-', # 从 stdin 读取
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p', # 浏览器兼容性必需
            '-preset', 'fast',
            '-crf', '23',
            '-loglevel', 'error', # 屏蔽日志，只显示错误
            output_path
        ]
        
        # 启动子进程
        # bufsize 设大一点有助于性能
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        return p
        
    def get_first_frame(self):
        """获取第一帧用于选择主体"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("无法读取视频第一帧")
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Convert to RGB for Gradio display
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def optimize_bbox_grabcut(self, frame, initial_bbox):
        """
        使用 OpenCV 原生 GrabCut 算法优化选中区域。
        """
        x, y, w, h = initial_bbox
        
        # 边界检查
        if w <= 0 or h <= 0:
            return initial_bbox
            
        # GrabCut 需要一定的边距，且框必须在图像内
        rect = (x, y, w, h)
        
        # 初始化掩码
        mask = np.zeros(frame.shape[:2], np.uint8)
        
        # GrabCut 内部使用的临时数组
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        try:
            # 运行 GrabCut (iterCount=5)
            cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            # 提取前景 (mask中 0和2是背景，1和3是前景)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # 查找前景的轮廓
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_cnt) < w * h * 0.1:
                    return initial_bbox
                    
                nx, ny, nw, nh = cv2.boundingRect(max_cnt)
                return (nx, ny, nw, nh)
            else:
                return initial_bbox
                
        except Exception as e:
            print(f"GrabCut Warning: {e}")
            return initial_bbox

    def process(self, initial_bbox, progress_callback=None, process_width=640):
        """
        执行 Dolly Zoom 处理
        initial_bbox: (x, y, w, h)
        progress_callback: function(current_frame, total_frames)
        process_width: 内部追踪使用的宽度，降低此值可显著提升速度 (默认 640)
        """
        # 重置视频流
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 初始化写入器 (使用 FFmpeg 管道)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        ffmpeg_writer = self._get_ffmpeg_writer(self.output_path, self.fps, self.width, self.height)
        
        # 初始化追踪器
        tracker = cv2.TrackerCSRT_create()
        
        ret, frame = self.cap.read()
        if not ret:
            return

        # 计算缩放比例 (如果视频本身很小，就不缩放)
        scale_factor = 1.0
        if self.width > process_width:
            scale_factor = process_width / self.width
            
        # 缩放第一帧用于初始化
        if scale_factor < 1.0:
            frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            # 缩放 bbox
            x, y, w, h = initial_bbox
            bbox_small = (
                int(x * scale_factor),
                int(y * scale_factor),
                int(w * scale_factor),
                int(h * scale_factor)
            )
        else:
            frame_small = frame
            bbox_small = initial_bbox

        # 优化选框 (在小图上进行，速度更快)
        bbox_small = self.optimize_bbox_grabcut(frame_small, bbox_small)
        tracker.init(frame_small, bbox_small)
        
        # 将优化后的 bbox 映射回原图，用于初始化 Kalman
        bbox = (
            int(bbox_small[0] / scale_factor),
            int(bbox_small[1] / scale_factor),
            int(bbox_small[2] / scale_factor),
            int(bbox_small[3] / scale_factor)
        )
        
        # 初始参数
        init_w, init_h = bbox[2], bbox[3]
        init_diagonal = np.sqrt(init_w**2 + init_h**2)
        
        # Kalman Filter 初始化
        kalman = cv2.KalmanFilter(6, 3)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        kalman.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        # 强平滑参数
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-5
        kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1
        
        # 初始状态
        init_cx = bbox[0] + bbox[2] / 2
        init_cy = bbox[1] + bbox[3] / 2
        kalman.statePre = np.array([[1.0], [init_cx], [init_cy], [0], [0], [0]], np.float32)
        kalman.statePost = np.array([[1.0], [init_cx], [init_cy], [0], [0], [0]], np.float32)

        # 滑动窗口平滑
        window_size = 30
        scale_history = []
        cx_history = []
        cy_history = []

        frame_idx = 0
        
        # 重置到第0帧开始循环
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 缩放当前帧用于追踪
            if scale_factor < 1.0:
                frame_small_curr = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            else:
                frame_small_curr = frame
                
            success, box_small = tracker.update(frame_small_curr)
            
            # 将追踪结果映射回原图坐标
            if success:
                box = (
                    box_small[0] / scale_factor,
                    box_small[1] / scale_factor,
                    box_small[2] / scale_factor,
                    box_small[3] / scale_factor
                )
            else:
                box = (0, 0, 0, 0) # 只是为了占位，下面的逻辑会处理 success=False
            
            # Kalman 预测
            prediction = kalman.predict()
            pred_scale = prediction[0, 0]
            pred_cx = prediction[1, 0]
            pred_cy = prediction[2, 0]
            
            if success:
                x, y, w, h = [int(v) for v in box]
                curr_cx = x + w / 2
                curr_cy = y + h / 2
                current_diagonal = np.sqrt(w**2 + h**2)
                
                if current_diagonal > 0:
                    raw_scale = init_diagonal / current_diagonal
                    raw_scale = max(0.1, min(raw_scale, 10.0))
                else:
                    raw_scale = pred_scale
                    
                measurement = np.array([[np.float32(raw_scale)], [np.float32(curr_cx)], [np.float32(curr_cy)]])
                estimated = kalman.correct(measurement)
                
                k_scale = estimated[0, 0]
                k_cx = estimated[1, 0]
                k_cy = estimated[2, 0]
            else:
                k_scale = pred_scale
                k_cx = pred_cx
                k_cy = pred_cy
                
            # 滑动窗口
            scale_history.append(k_scale)
            cx_history.append(k_cx)
            cy_history.append(k_cy)
            
            if len(scale_history) > window_size:
                scale_history.pop(0)
                cx_history.pop(0)
                cy_history.pop(0)
                
            smooth_scale = np.mean(scale_history)
            smooth_cx = np.mean(cx_history)
            smooth_cy = np.mean(cy_history)
            
            # 锁定到初始位置
            target_cx, target_cy = init_cx, init_cy
            
            M = np.float32([
                [smooth_scale, 0, target_cx - smooth_scale * smooth_cx],
                [0, smooth_scale, target_cy - smooth_scale * smooth_cy]
            ])
            
            frame_zoomed = cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REPLICATE)
            
            # out.write(frame_zoomed)
            # 写入到 FFmpeg stdin
            try:
                ffmpeg_writer.stdin.write(frame_zoomed.tobytes())
            except Exception as e:
                print(f"Error writing to ffmpeg: {e}")
                break
            
            frame_idx += 1
            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx, self.total_frames)
                
        self.cap.release()
        # out.release()
        
        # 关闭 FFmpeg 管道
        if ffmpeg_writer.stdin:
            ffmpeg_writer.stdin.close()
        ffmpeg_writer.wait()

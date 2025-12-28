import cv2
import numpy as np
import time
import os
import imageio_ffmpeg
import subprocess
from core import DollyZoomProcessor

class LiveDollyZoomApp:
    def __init__(self, camera_index=0, target_width=1280, target_height=720):
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        
        # 摄像头与画面参数
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps_cam = 30.0
        
        # 追踪与处理参数
        self.process_width = 480  # 追踪时的降采样宽度
        self.tracker = None
        self.selecting = False
        self.bbox = None
        self.initial_bbox_for_recording = None
        
        # Dolly Zoom 计算参数
        self.init_diagonal = 0
        self.init_cx = 0
        self.init_cy = 0
        
        # 平滑处理
        self.window_size = 10
        self.scale_history = []
        self.cx_history = []
        self.cy_history = []
        
        # 录制状态
        self.recording = False
        self.recorded_frames = []
        self.recording_start_time = 0
        
        # 帧率平滑
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps_buffer = []
        
        # 初始化摄像头
        self._init_camera()

    def _init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps_cam = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"摄像头分辨率: {self.width}x{self.height}, FPS: {self.fps_cam}")
        print("操作说明:")
        print("1. 按 's' 键暂停画面并选择追踪主体")
        print("2. 按 'space' 键开始/停止录制 (仅在追踪模式下有效)")
        print("3. 按 'r' 键重置追踪")
        print("4. 按 'q' 键退出")

    def _reset_tracker_state(self):
        self.tracker = None
        self.selecting = False
        self.recording = False
        self.recorded_frames = []
        self.scale_history = []
        self.cx_history = []
        self.cy_history = []

    def _init_tracking(self, frame, bbox):
        if bbox[2] <= 0 or bbox[3] <= 0:
            return False

        self.tracker = cv2.TrackerCSRT_create()
        self.initial_bbox_for_recording = bbox
        
        # 降采样初始化追踪器
        scale_factor = self.process_width / self.width
        frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        bbox_small = (
            int(bbox[0] * scale_factor),
            int(bbox[1] * scale_factor),
            int(bbox[2] * scale_factor),
            int(bbox[3] * scale_factor)
        )
        self.tracker.init(frame_small, bbox_small)
        
        # 计算初始几何信息
        init_w, init_h = bbox[2], bbox[3]
        self.init_diagonal = np.sqrt(init_w**2 + init_h**2)
        self.init_cx = bbox[0] + bbox[2] / 2
        self.init_cy = bbox[1] + bbox[3] / 2
        
        # 初始化历史队列
        self.scale_history = [1.0] * self.window_size
        self.cx_history = [self.init_cx] * self.window_size
        self.cy_history = [self.init_cy] * self.window_size
        
        return True

    def _update_tracking(self, frame):
        # 降采样更新追踪
        scale_factor = self.process_width / self.width
        frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        
        success, box_small = self.tracker.update(frame_small)
        
        is_too_fast = False
        if success:
            # 还原坐标
            x = box_small[0] / scale_factor
            y = box_small[1] / scale_factor
            w = box_small[2] / scale_factor
            h = box_small[3] / scale_factor
            
            curr_cx = x + w / 2
            curr_cy = y + h / 2
            curr_diagonal = np.sqrt(w**2 + h**2)
            
            # 检测移动速度
            if self.cx_history:
                prev_cx = self.cx_history[-1]
                prev_cy = self.cy_history[-1]
                # 计算两帧之间的位移距离
                displacement = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
                # 如果位移超过画面宽度的 3%
                if displacement > self.width * 0.03:
                    is_too_fast = True

            if curr_diagonal > 0:
                raw_scale = self.init_diagonal / curr_diagonal
                raw_scale = max(0.2, min(raw_scale, 5.0))
            else:
                raw_scale = 1.0
        else:
            # 丢失追踪时保持上一帧状态
            raw_scale = self.scale_history[-1] if self.scale_history else 1.0
            curr_cx = self.cx_history[-1] if self.cx_history else self.init_cx
            curr_cy = self.cy_history[-1] if self.cy_history else self.init_cy
            
        return success, raw_scale, curr_cx, curr_cy, is_too_fast

    def _apply_smoothing(self, raw_scale, curr_cx, curr_cy):
        self.scale_history.append(raw_scale)
        self.cx_history.append(curr_cx)
        self.cy_history.append(curr_cy)
        
        if len(self.scale_history) > self.window_size:
            self.scale_history.pop(0)
            self.cx_history.pop(0)
            self.cy_history.pop(0)
            
        smooth_scale = np.mean(self.scale_history)
        smooth_cx = np.mean(self.cx_history)
        smooth_cy = np.mean(self.cy_history)
        
        return smooth_scale, smooth_cx, smooth_cy

    def _warp_frame(self, frame, scale, cx, cy):
        M = np.float32([
            [scale, 0, self.init_cx - scale * cx],
            [0, scale, self.init_cy - scale * cy]
        ])
        return cv2.warpAffine(frame, M, (self.width, self.height), borderMode=cv2.BORDER_REPLICATE)

    def _draw_hud(self, frame, fps, scale=None, tracking_success=True, is_too_fast=False):
        # FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 追踪状态
        if self.tracker is None:
            cv2.putText(frame, "Press 's' to Select Target", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif not tracking_success:
            cv2.putText(frame, "Tracking Lost!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_too_fast:
            cv2.putText(frame, "Too Fast!", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2) # Orange
        elif scale is not None:
            cv2.putText(frame, f"Scale: {scale:.2f}x", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 录制状态
        if self.recording:
            cv2.circle(frame, (self.width - 50, 50), 20, (0, 0, 255), -1)
            rec_time = time.time() - self.recording_start_time
            cv2.putText(frame, f"REC {rec_time:.1f}s", (self.width - 150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        # 底部提示
        cv2.putText(frame, "Space: Record, Q: Quit, R: Reset", (20, self.height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _process_recording_post_production(self):
        if len(self.recorded_frames) <= 30:
            print("录制时间太短，已忽略。")
            return

        print(f"录制结束，共 {len(self.recorded_frames)} 帧。开始后期处理...")
        
        timestamp = int(time.time())
        # 原始视频保存到 temp 目录 (保留不删除)
        raw_output = f"temp/live_output_{timestamp}_raw.mp4"
        dolly_output = f"output/live_output_{timestamp}_dolly.mp4"
        
        if not os.path.exists("temp"):
            os.makedirs("temp")
        if not os.path.exists("output"):
            os.makedirs("output")
            
        # 1. 保存原始录像
        # 计算实际录制帧率，而不是使用摄像头标称帧率
        recording_duration = time.time() - self.recording_start_time
        if recording_duration > 0:
            real_fps = len(self.recorded_frames) / recording_duration
        else:
            real_fps = self.fps_cam
            
        print(f"录制统计: 时长 {recording_duration:.2f}s, 帧数 {len(self.recorded_frames)}, 实际帧率 {real_fps:.2f} FPS")
        
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out_writer = cv2.VideoWriter(raw_output, fourcc, real_fps, (self.width, self.height))
        
        # 使用 FFmpeg 管道写入 H.264
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(real_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            '-loglevel', 'error', # 屏蔽日志
            raw_output
        ]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        for f in self.recorded_frames:
            # out_writer.write(f)
            try:
                p.stdin.write(f.tobytes())
            except Exception as e:
                print(f"Error writing to ffmpeg live: {e}")
                break
                
        # out_writer.release()
        if p.stdin:
            p.stdin.close()
        p.wait()
        
        print("原始视频已保存，正在应用高质量 Dolly Zoom 效果...")
        cv2.destroyWindow("Live Preview")
        
        try:
            # 2. 调用 Core 处理
            processor = DollyZoomProcessor(raw_output, dolly_output)
            
            def console_progress(curr, total):
                if curr % 10 == 0:
                    print(f"Processing: {curr}/{total} frames...")
            
            processor.process(self.initial_bbox_for_recording, progress_callback=console_progress)
            
            print(f"\n✅ 处理完成！文件已保存至: {dolly_output}")
            print("按任意键继续预览...")
                
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        if self.cap is None:
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            t_start = time.time()
            
            # --- 模式处理 ---
            if self.selecting:
                # 选择模式
                bbox = cv2.selectROI("Live Preview", frame, fromCenter=False, showCrosshair=True)
                if self._init_tracking(frame, bbox):
                    self.selecting = False
                else:
                    self.selecting = False # 取消选择
                    
            elif self.tracker is not None:
                # 追踪模式
                if self.recording:
                    self.recorded_frames.append(frame.copy())
                
                success, raw_scale, curr_cx, curr_cy, is_too_fast = self._update_tracking(frame)
                smooth_scale, smooth_cx, smooth_cy = self._apply_smoothing(raw_scale, curr_cx, curr_cy)
                
                # 生成变焦预览画面
                display_frame = self._warp_frame(frame, smooth_scale, smooth_cx, smooth_cy)
                
                # 计算平滑 FPS
                self.new_frame_time = time.time()
                dt = self.new_frame_time - self.prev_frame_time
                self.prev_frame_time = self.new_frame_time
                self.fps_buffer.append(dt)
                if len(self.fps_buffer) > 10:
                    self.fps_buffer.pop(0)
                fps = 1.0 / (sum(self.fps_buffer) / len(self.fps_buffer) + 1e-6)
                
                self._draw_hud(display_frame, fps, smooth_scale, success, is_too_fast)
            else:
                # 待机模式
                self.new_frame_time = time.time()
                dt = self.new_frame_time - self.prev_frame_time
                self.prev_frame_time = self.new_frame_time
                
                curr_fps = 1 / dt if dt > 0 else 0
                self.fps_buffer.append(curr_fps)
                if len(self.fps_buffer) > 10:
                    self.fps_buffer.pop(0)
                avg_fps = sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0
                
                self._draw_hud(display_frame, avg_fps)

            # --- 显示与输入 ---
            cv2.imshow("Live Preview", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and self.tracker is None:
                self.selecting = True
            elif key == ord('r'):
                self._reset_tracker_state()
            elif key == ord(' '):
                if self.tracker is not None:
                    if not self.recording:
                        self.recording = True
                        self.recording_start_time = time.time()
                        self.recorded_frames = []
                        print("开始录制...")
                    else:
                        self.recording = False
                        self._process_recording_post_production()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiveDollyZoomApp()
    app.run()

import cv2
import numpy as np
import time
import socket
import struct
import threading
import os
import queue
from live_preview import LiveDollyZoomApp

class NetworkDollyZoom(LiveDollyZoomApp):
    def __init__(self, host='0.0.0.0', port=5000):
        # Initialize parent without camera
        self.camera_index = 0
        self.target_width = 1280
        self.target_height = 720
        
        # Override init to skip camera setup
        self.cap = None
        self.width = self.target_width
        self.height = self.target_height
        self.fps_cam = 30.0
        
        # Output FPS (Set to 30.0 for fixed frame rate, or None for real-time based on network speed)
        # If set to 30 and network is slow (e.g. 10fps), video will play 3x faster.
        self.output_fps = 30.0
        
        # Tracking params (same as parent)
        self.process_width = 480
        self.tracker = None
        self.selecting = False
        self.bbox = None
        self.initial_bbox_for_recording = None
        self.pending_bbox = None
        self.current_bbox = None
        
        # Dolly params
        self.init_diagonal = 0
        self.init_cx = 0
        self.init_cy = 0
        
        # Smoothing
        self.window_size = 10
        self.scale_history = []
        self.cx_history = []
        self.cy_history = []
        
        # FPS smoothing
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps_buffer = []
        
        # Recording (needed for HUD)
        self.recording = False
        self.recorded_frames = []
        self.recorded_bytes = [] # Store raw bytes for high performance recording
        self.recording_start_time = 0
        
        # Network
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")
        
        # Preview Queue
        self.preview_queue = queue.Queue(maxsize=2)
        self.send_lock = threading.Lock()
        self.running = True
        self.preview_thread = threading.Thread(target=self._preview_loop)
        self.preview_thread.daemon = True
        self.preview_thread.start()

    def _preview_loop(self):
        while self.running:
            try:
                # Get frame from queue (block briefly)
                try:
                    frame_data, rotation, conn = self.preview_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Decode frame
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                # Apply rotation
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Store frame dimensions
                self.height, self.width = frame.shape[:2]
                
                # Check for pending ROI/Tracking initialization
                if self.pending_bbox is not None:
                    print(f"Initializing tracking with pending bbox: {self.pending_bbox}")
                    self._init_tracking(frame, self.pending_bbox)
                    self.pending_bbox = None
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Encode response
                _, img_encoded = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                data = img_encoded.tobytes()
                
                # Send Type 1 (Image Response)
                try:
                    with self.send_lock:
                        conn.sendall(struct.pack('>I', 1))
                        conn.sendall(struct.pack('>I', len(data)))
                        conn.sendall(data)
                except Exception as e:
                    print(f"Error sending preview: {e}")
                    
            except Exception as e:
                print(f"Preview loop error: {e}")

    def process_frame(self, frame):
        # Resize if needed (optional, assuming client sends correct size or we handle it)
        self.height, self.width = frame.shape[:2]
        
        display_frame = frame.copy()
        
        # Logic from run() loop
        if self.selecting:
            # For network mode, we might need a different way to select.
            # For now, let's assume we auto-select center or wait for a command.
            # But to keep it simple, if 'selecting' is triggered (maybe by a flag), 
            # we initialize on the center object?
            # Let's just draw a rectangle and wait for a separate signal, 
            # or simplify: Always track center object if not tracking?
            pass

        if self.tracker is not None:
            success, raw_scale, curr_cx, curr_cy, is_too_fast = self._update_tracking(frame)
            smooth_scale, smooth_cx, smooth_cy = self._apply_smoothing(raw_scale, curr_cx, curr_cy)
            display_frame = self._warp_frame(frame, smooth_scale, smooth_cx, smooth_cy)
            
            # FPS Calculation
            self.new_frame_time = time.time()
            dt = self.new_frame_time - self.prev_frame_time
            self.prev_frame_time = self.new_frame_time
            if dt > 0:
                self.fps_buffer.append(dt)
                if len(self.fps_buffer) > 10: self.fps_buffer.pop(0)
                fps = 1.0 / (sum(self.fps_buffer) / len(self.fps_buffer))
            else:
                fps = 0
                
            self._draw_hud(display_frame, fps, smooth_scale, success, is_too_fast)
        else:
            # If not tracking, just return original frame with HUD
            cv2.putText(display_frame, "Waiting for tracking...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return display_frame

    def start(self):
        print(f"Server listening on {self.host}:{self.port}")
        
        try:
            while True:
                print("\nWaiting for connection...")
                conn, addr = self.server_socket.accept()
                print(f"Connected by {addr}")
                
                try:
                    while True:
                        # 1. Read message type (4 bytes)
                        type_data = self._recv_all(conn, 4)
                        if not type_data: 
                            print("Client disconnected (EOF)")
                            break
                        msg_type = struct.unpack('>I', type_data)[0]

                        if msg_type == 1: # Image
                            # 1.5 Read rotation (4 bytes)
                            rot_data = self._recv_all(conn, 4)
                            if not rot_data: break
                            rotation = struct.unpack('>I', rot_data)[0]

                            # 2. Read frame size (4 bytes)
                            size_data = self._recv_all(conn, 4)
                            if not size_data: break
                            size = struct.unpack('>I', size_data)[0]
                            
                            # 3. Read frame data
                            frame_data = self._recv_all(conn, size)
                            if not frame_data: break
                            
                            # 4. Store raw bytes if recording (High Performance)
                            if self.recording:
                                self.recorded_bytes.append((frame_data, rotation))
                            
                            # 5. Push to preview queue (Non-blocking)
                            try:
                                self.preview_queue.put_nowait((frame_data, rotation, conn))
                            except queue.Full:
                                pass # Drop frame if processing is too slow, to keep socket fast
                        
                        elif msg_type == 2: # ROI
                            # Read 4 floats (x, y, w, h) - 16 bytes
                            roi_data = self._recv_all(conn, 16)
                            if not roi_data: break
                            x, y, w, h = struct.unpack('>ffff', roi_data)
                            
                            print(f"Received ROI: x={x}, y={y}, w={w}, h={h}")
                            
                            if hasattr(self, 'width') and hasattr(self, 'height'):
                                # Denormalize
                                abs_x = int(x * self.width)
                                abs_y = int(y * self.height)
                                abs_w = int(w * self.width)
                                abs_h = int(h * self.height)
                                
                                bbox = (abs_x, abs_y, abs_w, abs_h)
                                self.pending_bbox = bbox
                                print(f"Pending ROI set: {bbox}")
                        
                        elif msg_type == 3: # Reset
                            print("Reset command received")
                            self._reset_tracker_state()
                            self.pending_bbox = None
                        
                        elif msg_type == 4: # Start Recording
                            # Allow recording if tracker is ready OR if we have a pending bbox
                            if (self.tracker is not None or self.pending_bbox is not None) and not self.recording:
                                print("Start Recording command received")
                                self.recording = True
                                self.recording_start_time = time.time()
                                self.recorded_frames = [] # Clear legacy frames
                                self.recorded_bytes = [] # Clear byte buffer
                                # self.output_fps = None # Reset FPS for new recording
                                
                                # Update initial bbox for recording
                                if self.current_bbox is not None:
                                    self.initial_bbox_for_recording = self.current_bbox
                                    print(f"Recording started. Initial bbox updated to: {self.initial_bbox_for_recording}")
                                elif self.pending_bbox is not None:
                                    print(f"Recording started with pending bbox. Initial bbox will be set on next frame.")
                                else:
                                    print(f"Recording started. Using existing initial bbox: {self.initial_bbox_for_recording}")
                            else:
                                print(f"Ignored Start Recording: Tracker={self.tracker is not None}, Pending={self.pending_bbox is not None}, Recording={self.recording}")

                        elif msg_type == 5: # Stop Recording
                            if self.recording:
                                print("Stop Recording command received")
                                self.recording = False
                                stop_time = time.time()
                                
                                raw_count = len(self.recorded_bytes)
                                duration = stop_time - self.recording_start_time
                                if duration > 0:
                                    actual_fps = raw_count / duration
                                    print(f"Recording stopped. Duration: {duration:.2f}s, Frames: {raw_count}, Calculated FPS: {actual_fps:.2f}")
                                    # User requested fixed 30 FPS output
                                    # self.output_fps = actual_fps
                                else:
                                    # self.output_fps = 30.0 # Fallback
                                    pass
                                
                                print(f"Recorded {raw_count} frames (raw bytes). Decoding...")
                                
                                # Decode all recorded frames
                                self.recorded_frames = []
                                for data, rot in self.recorded_bytes:
                                    nparr = np.frombuffer(data, np.uint8)
                                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        # Apply rotation
                                        if rot == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                                        elif rot == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
                                        elif rot == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                        self.recorded_frames.append(frame)
                                
                                # Process recording (post-production)
                                output_path = self._process_recording_post_production()
                                
                                if output_path and os.path.exists(output_path):
                                    print(f"Sending video file: {output_path}")
                                    with open(output_path, 'rb') as f:
                                        video_data = f.read()
                                        
                                    # Send Type 6 (File Transfer) safely
                                    with self.send_lock:
                                        conn.sendall(struct.pack('>I', 6))
                                        conn.sendall(struct.pack('>I', len(video_data)))
                                        conn.sendall(video_data)
                                    print("Video file sent successfully")
                                else:
                                    print("Processing failed or no file generated.")
                                    # Send Type 6 with size 0 to indicate failure
                                    with self.send_lock:
                                        conn.sendall(struct.pack('>I', 6))
                                        conn.sendall(struct.pack('>I', 0))
                                    
                            else:
                                print("Ignored Stop Recording: Not recording")

                except Exception as e:
                    print(f"Connection Error: {e}")
                finally:
                    conn.close()
                    print(f"Connection closed for {addr}")

        except KeyboardInterrupt:
            print("\nServer stopping...")
        finally:
            self.server_socket.close()

    def _recv_all(self, sock, n):
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

if __name__ == "__main__":
    # Get local IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Server IP: {local_ip}")
    
    server = NetworkDollyZoom(host='0.0.0.0', port=6000)
    server.start()

import cv2
import numpy as np
import time

def live_dolly_zoom():
    # 0 通常是默认摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置较高的分辨率 (取决于摄像头支持)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"摄像头分辨率: {width}x{height}")
    print("操作说明:")
    print("1. 按 's' 键暂停画面并选择追踪主体")
    print("2. 按 'q' 键退出")
    
    tracker = None
    init_diagonal = 0
    init_cx, init_cy = 0, 0
    
    # 简单的平滑队列
    scale_history = []
    cx_history = []
    cy_history = []
    window_size = 10  # 实时预览不需要太长的平滑窗口，否则会有延迟感
    
    # 追踪处理分辨率 (降低以提升FPS)
    process_width = 480 
    
    selecting = False
    bbox = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 镜像翻转，符合自拍预览习惯
        frame = cv2.flip(frame, 1)
        
        display_frame = frame.copy()
        
        if tracker is None:
            # 待机模式
            cv2.putText(display_frame, "Press 's' to Select Target", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 如果处于选择模式
            if selecting:
                # 弹窗选择 (selectROI 会阻塞，所以逻辑上是暂停的)
                bbox = cv2.selectROI("Live Preview", frame, fromCenter=False, showCrosshair=True)
                # selectROI 可能会在按下回车后关闭窗口，我们需要重新处理
                
                if bbox[2] > 0 and bbox[3] > 0:
                    # 初始化追踪器
                    # 使用 KCF 速度更快，适合实时；CSRT 精度高但慢
                    # 这里为了实时性，如果 CSRT 卡顿，可以换 cv2.TrackerKCF_create()
                    tracker = cv2.TrackerCSRT_create() 
                    
                    # 缩放初始化
                    scale_factor = process_width / width
                    frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                    bbox_small = (
                        int(bbox[0] * scale_factor),
                        int(bbox[1] * scale_factor),
                        int(bbox[2] * scale_factor),
                        int(bbox[3] * scale_factor)
                    )
                    
                    tracker.init(frame_small, bbox_small)
                    
                    init_w, init_h = bbox[2], bbox[3]
                    init_diagonal = np.sqrt(init_w**2 + init_h**2)
                    init_cx = bbox[0] + bbox[2] / 2
                    init_cy = bbox[1] + bbox[3] / 2
                    
                    # 清空历史
                    scale_history = [1.0] * window_size
                    cx_history = [init_cx] * window_size
                    cy_history = [init_cy] * window_size
                    
                    selecting = False
                else:
                    selecting = False # 取消选择
                    
        else:
            # 追踪模式
            t_start = time.time()
            
            # 降采样追踪
            scale_factor = process_width / width
            frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            
            success, box_small = tracker.update(frame_small)
            
            if success:
                # 还原坐标
                box = (
                    box_small[0] / scale_factor,
                    box_small[1] / scale_factor,
                    box_small[2] / scale_factor,
                    box_small[3] / scale_factor
                )
                
                x, y, w, h = box
                curr_cx = x + w / 2
                curr_cy = y + h / 2
                curr_diagonal = np.sqrt(w**2 + h**2)
                
                # 计算缩放
                if curr_diagonal > 0:
                    raw_scale = init_diagonal / curr_diagonal
                    # 限制缩放范围，防止过度拉伸
                    raw_scale = max(0.2, min(raw_scale, 5.0))
                else:
                    raw_scale = 1.0
                    
                # 绘制追踪框 (可选，调试用)
                # p1 = (int(x), int(y))
                # p2 = (int(x + w), int(y + h))
                # cv2.rectangle(display_frame, p1, p2, (255, 0, 0), 2, 1)
                
            else:
                # 丢失追踪，保持最后状态或重置
                raw_scale = scale_history[-1]
                curr_cx = cx_history[-1]
                curr_cy = cy_history[-1]
                cv2.putText(display_frame, "Tracking Lost!", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 平滑处理
            scale_history.append(raw_scale)
            cx_history.append(curr_cx)
            cy_history.append(curr_cy)
            
            if len(scale_history) > window_size:
                scale_history.pop(0)
                cx_history.pop(0)
                cy_history.pop(0)
                
            smooth_scale = np.mean(scale_history)
            smooth_cx = np.mean(cx_history)
            smooth_cy = np.mean(cy_history)
            
            # 应用变换
            # 目标是保持主体在画面位置和大小不变
            # M = [scale, 0, target_cx - scale * current_cx]
            #     [0, scale, target_cy - scale * current_cy]
            
            # 这里我们简化逻辑：
            # 1. 缩放画面
            # 2. 平移画面，使得当前主体中心 (smooth_cx, smooth_cy) 对齐到 初始主体中心 (init_cx, init_cy)
            
            M = np.float32([
                [smooth_scale, 0, init_cx - smooth_scale * smooth_cx],
                [0, smooth_scale, init_cy - smooth_scale * smooth_cy]
            ])
            
            # 实时变焦效果
            frame_zoomed = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
            
            # 计算FPS
            fps = 1 / (time.time() - t_start)
            cv2.putText(frame_zoomed, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_zoomed, f"Scale: {smooth_scale:.2f}x", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_zoomed, "Press 'q' to Quit, 'r' to Reset", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            display_frame = frame_zoomed

        cv2.imshow("Live Preview", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and tracker is None:
            selecting = True
        elif key == ord('r'):
            tracker = None
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_dolly_zoom()

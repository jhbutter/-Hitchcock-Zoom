import os
import cv2
import argparse
from core import DollyZoomProcessor

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Hitchcock Dolly Zoom Effect Generator")
    parser.add_argument("--input", "-i", default="input/input_dolly.mp4", help="Path to input video")
    parser.add_argument("--output", "-o", default="output/output_hitchcock.mp4", help="Path to output video")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    
    # 确保路径是绝对路径或正确的相对路径
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)

    # Check input
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    # Ensure output dir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        processor = DollyZoomProcessor(input_path, output_path)
        
        # Get first frame for selection
        # Note: get_first_frame returns RGB (for Gradio), convert to BGR for OpenCV GUI
        frame_rgb = processor.get_first_frame()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        print("\n=== 操作说明 ===")
        print("1. 弹出的窗口中，请用鼠标框选你要追踪的主体")
        print("2. 按下 'SPACE' 或 'ENTER' 确认选区")
        print("3. 按下 'c' 取消")
        print("================\n")
        
        bbox = cv2.selectROI("Select Target", frame_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Target")
        
        # selectROI return (0,0,0,0) if cancelled
        if bbox == (0, 0, 0, 0) or bbox[2] == 0 or bbox[3] == 0:
            print("未选择有效区域，程序退出")
            return

        print(f"初始选区: {bbox}")
        
        # Run processing
        def progress_callback(curr, total):
            percent = (curr / total) * 100
            bar_length = 30
            filled_length = int(bar_length * curr // total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rProgress: |{bar}| {percent:.1f}% ({curr}/{total})", end="")

        print("开始处理...")
        processor.process(bbox, progress_callback=progress_callback)
        print("\n\n处理完成！")
        print(f"输出文件已保存至: {output_path}")

    except Exception as e:
        print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main()

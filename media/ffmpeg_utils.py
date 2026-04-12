import subprocess
import cv2
import numpy as np
import os

class RTSPStreamer:
    """
    RTSP 影格推流器
    
    將處理後的 OpenCV 影格透過 FFmpeg 推送至 MediaMTX。
    """
    def __init__(self, rtsp_url: str = "rtsp://localhost:8554/detected", fps: int = 15, width: int = 640, height: int = 480):
        self.rtsp_url = rtsp_url.replace("localhost", "127.0.0.1")
        self.fps = fps
        self.width = width
        self.height = height
        self.process = None

    def start(self):
        """啟動 FFmpeg 子進程進行推流"""
        # 使用更穩定的推流參數
        command = [
            'ffmpeg',
            '-y',
            '-v', 'error',               # 只顯示錯誤
            '-thread_queue_size', '1024', # 進一步增加緩衝
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{self.width}x{self.height}",
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-bf', '0',                  # 關閉 B-frames 降低延遲與複雜度
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            self.rtsp_url
        ]
        
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        import time
        time.sleep(0.5) # 給予 FFmpeg 握手時間
        print(f"🚀 RTSP Streamer started: {self.rtsp_url}")

    def push_frame(self, frame: np.ndarray):
        """將影格寫入 FFmpeg stdin"""
        if self.process is None or self.process.poll() is not None:
            print("🔄 [RTSPStreamer] Restarting FFmpeg process...")
            self.start()
            
        try:
            # 確保影格尺寸正確
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except (IOError, BrokenPipeError) as e:
            print(f"⚠️ [RTSPStreamer] Broken pipe detected, will restart on next frame: {e}")
            self.stop()
        except Exception as e:
            print(f"❌ [RTSPStreamer] Unexpected error: {e}")

    def stop(self):
        """停止推流"""
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process = None

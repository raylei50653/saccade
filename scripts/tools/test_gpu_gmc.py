import torch
import numpy as np
import cv2
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
build_path = project_root / "build"
if build_path.exists():
    sys.path.insert(0, str(build_path))

try:
    from saccade_tracking_ext import GMC as CppGMC
except ImportError:
    print("Cannot import saccade_tracking_ext. Build the project first.")
    sys.exit(1)

def test_gmc():
    device = torch.device("cuda")
    w, h = 1920, 1080
    downscale = 8
    
    # Create a simple image: black with a white square
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (w//4, h//4), (3*w//4, 3*h//4), (255, 255, 255), -1)
    
    # Convert to CHW float and ensure contiguous
    frame1 = torch.from_numpy(img).permute(2, 0, 1).float().to(device).contiguous()
    
    # Shift image by (dx, dy) - use integers to avoid interpolation artifacts for now
    dx_true, dy_true = 16.0, -8.0
    M = np.float32([[1, 0, dx_true], [0, 1, dy_true]])
    img2 = cv2.warpAffine(img, M, (w, h))
    frame2 = torch.from_numpy(img2).permute(2, 0, 1).float().to(device).contiguous()
    
    gmc = CppGMC(downscale=downscale)
    stream = torch.cuda.current_stream().cuda_stream
    
    # First frame to initialize
    _ = gmc.estimate(frame1.data_ptr(), w, h, stream, True)
    
    # Second frame to estimate motion
    warp = gmc.estimate(frame2.data_ptr(), w, h, stream, True)
    
    if warp is None:
        print("GMC returned None")
    else:
        dx_est = warp[2]
        dy_est = warp[5]
        print(f"True shift: ({dx_true}, {dy_true})")
        print(f"Est shift:  ({dx_est:.4f}, {dy_est:.4f})")
        print(f"Error:      ({dx_est - dx_true:.4f}, {dy_est - dy_true:.4f})")

if __name__ == "__main__":
    test_gmc()

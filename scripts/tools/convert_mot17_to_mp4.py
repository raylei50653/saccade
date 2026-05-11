import subprocess
from pathlib import Path
import configparser
from concurrent.futures import ProcessPoolExecutor

def convert_single_sequence(seq_path: Path, out_root: Path):
    seq_name = seq_path.name
    img_dir = seq_path / "img1"
    info_file = seq_path / "seqinfo.ini"
    
    # 讀取 frame rate
    fps = 30
    if info_file.exists():
        config = configparser.ConfigParser()
        config.read(info_file)
        fps = config.getint("Sequence", "frameRate", fallback=30)
    
    output_file = out_root / f"{seq_name}.mp4"
    
    print(f"🎬 [Start] {seq_name} -> {output_file} ({fps} FPS)")
    
    # ffmpeg 指令：使用 h264_nvenc 進行 GPU 加速編碼
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(img_dir / "%06d.jpg"),
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-preset", "p4", # 稍微調低一點以支援更高並行度
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "18",
        "-b:v", "0",
        str(output_file)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ [Done] {seq_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [Failed] {seq_name}: {e.stderr.decode()}")
        return False

def convert_sequences_parallel(root_dir: str, output_dir: str, max_workers: int = 4):
    root = Path(root_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 搜尋所有包含 img1 的序列目錄
    sequences = [p.parent for p in root.glob("**/img1")]
    
    print(f"🚀 Starting parallel conversion with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_single_sequence, seq, out_root) for seq in sequences]
        results = [f.result() for f in futures]
    
    success_count = sum(1 for r in results if r)
    print(f"\n🏁 Completed: {success_count}/{len(sequences)} sequences successful.")

if __name__ == "__main__":
    # 預設路徑
    mot17_root = "datasets/MOT17/train"
    output_path = "datasets/MOT17_videos"
    
    if not Path(mot17_root).exists():
        print(f"⚠️  MOT17 root not found at {mot17_root}")
    else:
        # 根據 GPU 能力調整並行數，一般家用卡建議 2-4，專業卡可更高
        convert_sequences_parallel(mot17_root, output_path, max_workers=4)

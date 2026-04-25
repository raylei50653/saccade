from pathlib import Path


def convert_mot17_to_yolo(data_root):
    data_root = Path(data_root)
    for seq in (data_root / "train").iterdir():
        if not seq.name.startswith("MOT17-"):
            continue

        # 建立 labels 目錄
        labels_dir = seq / "labels"
        labels_dir.mkdir(exist_ok=True)

        # 讀取 seqinfo 獲取解析度
        import configparser

        config = configparser.ConfigParser()
        config.read(seq / "seqinfo.ini")
        w = int(config.get("Sequence", "imWidth"))
        h = int(config.get("Sequence", "imHeight"))

        # 讀取 gt.txt
        gt_file = seq / "gt/gt.txt"
        if not gt_file.exists():
            continue

        # MOT 格式: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <class>, <visibility>
        with open(gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if int(parts[7]) != 1:
                    continue  # 只保留行人 (class 1)

                frame = int(parts[0])
                # YOLO 格式: <class> <x_center> <y_center> <width> <height> (normalized)
                bw = float(parts[4])
                bh = float(parts[5])
                bx = float(parts[2]) + bw / 2.0
                by = float(parts[3]) + bh / 2.0

                yolo_line = f"0 {bx / w:.6f} {by / h:.6f} {bw / w:.6f} {bh / h:.6f}\n"

                with open(labels_dir / f"{frame:06d}.txt", "a") as out:
                    out.write(yolo_line)
        print(f"✅ Converted {seq.name}")


if __name__ == "__main__":
    convert_mot17_to_yolo("datasets/MOT17")

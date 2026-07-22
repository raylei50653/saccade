"""Unit tests for the temporal-YOLO MOT17 clip dataset loader (perception.temporal_yolo.dataset)."""

# scope: detection
# function: behavior
# lifecycle: active

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from saccade.perception.temporal_yolo.dataset import MOT17TemporalClip, _load_and_resize

# This test mocks the module-level `open`/`csv`/`Path` helpers but the dataset
# reads gt.txt via `Path.open()`, which is not mocked — so it needs the real
# MOT17 dataset on disk. Skip when absent (e.g. CI) instead of FileNotFoundError.
_GT = Path("datasets/MOT17/train/MOT17-02-SDP/gt/gt.txt")
skip_no_dataset = pytest.mark.skipif(
    not _GT.exists(), reason="MOT17 dataset not found (e.g. CI)"
)


@skip_no_dataset
@patch("saccade.perception.temporal_yolo.dataset.Path.exists")
@patch("saccade.perception.temporal_yolo.dataset.configparser.ConfigParser")
@patch("saccade.perception.temporal_yolo.dataset.csv.reader")
@patch("saccade.perception.temporal_yolo.dataset.open")
def test_dataset_letterbox_init(mock_open, mock_csv, mock_config, mock_exists):
    # Mock filesystem paths and existence
    mock_exists.return_value = True

    # Mock seqinfo.ini values
    mock_parser = MagicMock()
    mock_parser.__getitem__.return_value = {"imHeight": "1080", "imWidth": "1920"}
    mock_config.return_value = mock_parser

    # Mock csv reader returning empty GT list
    mock_csv.return_value = []

    # Mock sequences directory listing
    with patch("saccade.perception.temporal_yolo.dataset.Path.iterdir") as mock_iterdir:
        mock_seq = MagicMock()
        mock_seq.is_dir.return_value = True
        mock_seq.name = "MOT17-02-SDP"
        mock_iterdir.return_value = [mock_seq]

        with patch("saccade.perception.temporal_yolo.dataset.Path.glob") as mock_glob:
            mock_glob.return_value = [
                Path("datasets/MOT17/train/MOT17-02-SDP/img1/000001.jpg")
            ] * 10

            # Build dataset under img_size=640
            dataset = MOT17TemporalClip(
                data_root="datasets/MOT17",
                clip_len=4,
                img_size=640,
                preload_to_ram=False,
                use_letterbox=True,
            )

            assert "MOT17-02-SDP" in dataset._scale_hw
            scale, pad_h, pad_w = dataset._scale_hw["MOT17-02-SDP"]

            # Under orig_h=1080, orig_w=1920, target_size=640:
            # scale = min(640/1080, 640/1920) = 640/1920 = 1/3
            # new_h = round(1080 * 1/3) = 360
            # new_w = round(1920 * 1/3) = 640
            # pad_h = (640 - 360) // 2 = 140
            # pad_w = (640 - 640) // 2 = 0
            assert abs(scale - 0.33333) < 1e-4
            assert pad_h == 140
            assert pad_w == 0


@patch("torchvision.io.read_image")
def test_load_and_resize_letterbox(mock_read_image):
    # Mock input image as 1080p rectangular (3, 1080, 1920)
    mock_read_image.return_value = torch.ones(3, 1080, 1920, dtype=torch.uint8) * 255

    # Run _load_and_resize
    img_size = 640
    scale = 640 / 1920
    pad_h = 140
    pad_w = 0

    img_padded = _load_and_resize(
        Path("dummy.jpg"),
        img_size,
        use_letterbox=True,
        scale_or_h=scale,
        pad_h_or_w=pad_h,
        pad_w=pad_w,
    )

    # Padded image must be square (3, 640, 640)
    assert img_padded.shape == (3, 640, 640)

    # Pixels in the pad region (e.g. top rows 0 to 139) should be padded with constant 114
    assert torch.all(img_padded[:, 0:139, :] == 114)
    # Pixels in the center region (e.g. rows 140 to 499) should be the original 255
    assert torch.all(img_padded[:, 140:499, :] == 255)
    # Pixels in the bottom pad region (rows 500 to 639) should be padded with constant 114
    assert torch.all(img_padded[:, 500:639, :] == 114)


def test_metadata_only_dataset_does_not_decode_images(tmp_path):
    seq_dir = tmp_path / "train" / "MOT17-04-SDP"
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir()
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nimHeight=1080\nimWidth=1920\n",
        encoding="utf-8",
    )
    (gt_dir / "gt.txt").write_text(
        "1,1,10,20,30,40,1,1,1.0\n",
        encoding="utf-8",
    )
    for frame_id in range(1, 5):
        (img_dir / f"{frame_id:06d}.jpg").touch()

    dataset = MOT17TemporalClip(
        data_root=tmp_path,
        clip_len=4,
        stride=4,
        seqs=["MOT17-04-SDP"],
        preload_to_ram=False,
        load_images=False,
    )
    with patch("torchvision.io.read_image") as read_image:
        sample = dataset[0]

    read_image.assert_not_called()
    assert sample["frames"].shape == (4, 0, 0, 0)
    assert sample["frame_ids"] == [1, 2, 3, 4]

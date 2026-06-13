import torch
import torchvision.io as tv_io

from saccade.perception.temporal_yolo.dataset import _load_detail_view


def test_detail_view_preserves_aspect_ratio_and_pads_bottom(tmp_path):
    image = torch.zeros(3, 20, 40, dtype=torch.uint8)
    image[:, :, :20] = 200
    path = tmp_path / "frame.jpg"
    tv_io.write_jpeg(image, str(path), quality=100)

    detail, valid_hw = _load_detail_view(path, (32, 32))

    assert detail.shape == (3, 32, 32)
    assert valid_hw.tolist() == [16, 32]
    assert detail[:, :16].float().mean() > 80
    assert torch.all(detail[:, 16:] == 114)

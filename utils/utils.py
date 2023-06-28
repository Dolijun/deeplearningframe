import torch
import cv2
import numpy as np
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from matplotlib.backends.backend_agg import FigureCanvasAgg


def img_from_canvas(canvas: 'FigureCanvasAgg') -> np.ndarray:
    """
    Get RGB image from ``FigureCanvasAgg``.
    Args:
        canvas (FigureCanvasAgg): The canvas to get image.
    Returns:
        np.ndarray: the output of image in RGB.
    """
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype=np.uint8)
    buffer = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(buffer, [3], axis=2)
    return rgb.astype(np.uint8)


def convert_overlay_heatmap(feat_map: Union[np.ndarray, torch.Tensor],
                            img: Optional[np.ndarray],
                            alpha: float = 0.5) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be "RGB". Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3 and feat_map.shape[0] in [1, 3]), f"feat_map.ndim is illegal"
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = np.transpose(feat_map, (1, 2, 0))

    norm_img = np.zeros_like(feat_map)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img


# for sed
# 把二进制真值转化为 mask 真值
def binary_file_to_channel_masks(bin_file, height, width, channels, ignore_pixel_id_map=(31, 255)):
    # read file
    edge_bin = np.fromfile(bin_file, dtype=np.uint32)
    edge_bin = edge_bin.reshape(height, width)
    edge_mask = np.zeros((channels, height, width), dtype=np.float32)
    ignore_edge_mask = (edge_bin & 1 << ignore_pixel_id_map[0]) > 0
    for c in range(channels):
        mask = (edge_bin & 1 << c) > 1
        edge_mask[c, :, :] = mask
        edge_mask[ignore_edge_mask] = ignore_pixel_id_map[1]
    return edge_mask.transpose((1, 2, 0))

import numpy as np

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
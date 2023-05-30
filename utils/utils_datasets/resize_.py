import cv2
import os
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--image_path", help="输入图片文件夹", required=True)
parser.add_argument("--save_path", help="图片保存文件夹", required=True)
parser.add_argument("--fix_short", help="是否只保存最短边", action="store_true")
parser.add_argument("-s", "--short_edge", help="target short length", type=int)
parser.add_argument("-h", "--height", help="target height", type=int)
parser.add_argument("-w", "--width", help="target width", type=int)
parser.add_argument("-d", "--dpi", help="target dpi", type=float, default=96.0)
parser.add_argument("-q", "--quality", help="target quality", type=float, default=95)


def resize(img_path, save_path, x=256, y=256, quality=95, dpi=(72.0, 72.0)):
    # 设置图像的输入、输出、resize大小、质量和dpi值
    img_name = os.listdir(img_path)
    for name in img_name:
        in_name = os.path.join(img_name, name)
        out_name = os.path.join(save_path, name)

        im = cv2.imread(in_name)
        assert im is not None, "imread None!"

        im_resize = cv2.resize(im, (x, y))
        im_dpi = Image.fromarray(cv2.cvtColor(im_resize, cv2.COLOR_BGR2RGB))
        im_dpi.save(out_name, quality=quality, dpi=dpi)


def resize_fix_short_length(im_path, save_path, short_size=256, quality=96, dpi=(72.0, 72.0)):
    im_names = os.listdir(im_path)

    for im_name in im_names:
        in_name = os.path.join(im_path, im_name)
        out_name = os.path.join(save_path, im_name)
        out_dir = os.path.dirname(out_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 读取图像
        im = cv2.imread(in_name)
        assert im is not None, "imread None"
        h, w, _ = im.shape
        if h > w:
            w_new = short_size
            h_new = int(short_size * h * 1.0 / w)
        else:
            h_new = short_size
            w_new = int(short_size * w * 1.0 / h)
        im_resize = cv2.resize(im, (w_new, h_new))
        im_dpi = Image.fromarray(cv2.cvtColor(im_resize, cv2.COLOR_BGR2RGB))
        im_dpi.save(out_name, quality, dpi=dpi)


def main(args):
    assert os.path.exists(args.image_path), "image_path not exists!"
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    image_path = args.image_path
    save_path = args.save_path
    quality = args.quality
    dpi = (args.dpi, args.api)
    if args.fix_short:
        assert args.short_edge is not None, "no short edge length!"
        resize_fix_short_length(image_path, save_path, args.short_edge, quality, dpi)
    else:
        assert args.height is not None and args.width is not None, "no height or width"
        resize(image_path, save_path, args.width, args.height, quality, dpi)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

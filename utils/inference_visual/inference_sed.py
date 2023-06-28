import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import re
from utils.datasets.sed_utils import binary_file_to_channel_masks
from dataset.city_datasets import DataList
import scipy.io as sci
import tqdm

class ActivationWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activation = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )

    def save_activation(self, module, input, output):
        self.activation.append(output)

    def __call__(self, inputs):

        ## todo: model inference
        self.activation = []
        results = self.model(inputs)
        return results, self.activation

    def release(self):
        for handle in self.handles:
            handle.remove()


def get_visual_colors(dataset):
    # RGB !!!
    if dataset == 'cityscapes':
        colors = [[128, 64, 128],
                  [244, 35, 232],
                  [70, 70, 70],
                  [102, 102, 156],
                  [190, 153, 153],
                  [153, 153, 153],
                  [250, 170, 30],
                  [220, 220, 0],
                  [107, 142, 35],
                  [152, 251, 152],
                  [70, 130, 180],
                  [220, 20, 60],
                  [255, 0, 0],
                  [0, 0, 142],
                  [0, 0, 70],
                  [0, 60, 100],
                  [0, 80, 100],
                  [0, 0, 230],
                  [119, 11, 32]]
    else:
        assert dataset == 'sbd'
        colors = [[128, 0, 0],
                  [0, 128, 0],
                  [128, 128, 0],
                  [0, 0, 128],
                  [128, 0, 128],
                  [0, 128, 128],
                  [128, 128, 128],
                  [64, 0, 0],
                  [192, 0, 0],
                  [64, 128, 0],
                  [192, 128, 0],
                  [64, 0, 128],
                  [192, 0, 128],
                  [64, 128, 128],
                  [192, 128, 128],
                  [0, 64, 0],
                  [128, 64, 0],
                  [0, 192, 0],
                  [128, 192, 0],
                  [0, 64, 128],
                  [255, 255, 255]]

    return colors


# load thresh from matlab result
# .mat should in the path
def load_thresh(path, cls):
    thresh = []
    for c in range(cls):
        filename = f"class_{c + 1}.mat"
        file_path = os.path.join(path, filename)
        if os.path.exists(file_path):
            result = sci.loadmat(file_path)
            thresh.append(result['result_cls'][1][0][0][0])
        else:
            raise ValueError(f"No {filename} in dir {path}")
    return thresh


# read flist from file
def default_flist_reader(flist):
    imlist = []
    with open(flist, "r") as rf:
        for line in rf.readlines():
            splitted = line.strip().split()
            if len(splitted) == 2:
                impath, imlabel = splitted
            elif len(splitted) == 1:
                impath, imlabel = splitted[0], None
            else:
                raise ValueError("flist line value error")
            impath = impath.strip("../")
            imseg = imlabel.replace("edge.bin", "trainIds.png")
            imlist.append(impath, imlabel, imseg)
    return imlist


# predict image
def image_predict(model_list, img):
    '''
    :param model_list: default(backbone, net)
    :param img: BGR - mean_value, torch.tensor, 3*H*W
    :return:
        sed_pred: torch.tensor, 1*cls*H*W, no sigmoid
        seg_pred: torch.tensor, 1*cls*H*W, no argmax
        edge_pred: torch.tensor, 1*1*H*W, no sigmoid
    '''
    # checking img.ndim
    if img.ndim == 3:
        img = img[None]
    if img.ndim != 4:
        raise ValueError("img ndim illegal")

    with torch.no_grad():
        # custom predict
        backbone = model_list[0]
        net = model_list[1]
        features = backbone(img)
        out_preds = net(img, *features)

        sed_pred = torch.sigmoid(out_preds[0])
        seg_pred = out_preds[1]
        edge_pred = torch.sigmoid(out_preds[2])

    # custom the return
    return sed_pred, seg_pred, edge_pred


# predict image patches
def patch_predict(model_list, img, n_classes, patch_h, patch_w, step_size_y, step_size_x, pad):
    '''
    :param model_list: default(backbone, net)
    :param img: BGR - mean_value, torch.Tensor, 3*H*W
    :param patch_h: int
    :param patch_w: int
    :param step_size_y: int
    :param step_size_x: int
    :param pad: int, pad is for overlaped inference
    :return: pred result
    '''
    # checking img ndim
    if img.ndim == 3:
        img = img[None]
    if img.ndim != 4:
        raise ValueError(f"img ndim illegal, expect ndim in [3, 4], but got img.ndim={img.ndim}")
    n, c, h, w = img.shape
    # padding image
    assert (w - patch_w + 0.0) % step_size_x == 0, "padding image width must be divided by step_size_x"
    assert (h - patch_h + 0.0) % step_size_y == 0, "padding image height must be divided by step_size_y"
    step_num_x = int((w - patch_w + 0.0) / step_size_x) + 1
    step_num_y = int((h - patch_h + 0.0) / step_size_y) + 1

    # for overlaped inference
    img = F.pad(img, (pad, pad, pad, pad), 'constant', 0)

    # init temp result
    sed_out = torch.zeros(n, n_classes, h + 2 * pad, w + 2 * pad).cuda()
    seg_out = torch.zeros(n, n_classes, h + 2 * pad, w + 2 * pad).cuda()
    edge_out = torch.zeros(n, 1, h + 2 * pad, w + 2 * pad).cuda()
    mat_count = torch.zeros(n, 1, h + 2 * pad, w + 2 * pad).cuda()

    # do patch and merge
    for i in range(step_num_y):
        offset_y = i * step_size_y
        for j in range(step_num_x):
            offset_x = j * step_size_x
            patch_in = img[:, :, offset_y:offset_y + patch_h + 2 * pad, offset_x:offset_x + patch_w + 2 * pad]
            sed_pred, seg_pred, edge_pred = image_predict(model_list, patch_in)

            sed_out[:, :, offset_y:offset_y + patch_h + 2 * pad, offset_x:offset_x + patch_w + 2 * pad] += \
                sed_pred
            seg_out[:, :, offset_y:offset_y + patch_h + 2 * pad, offset_x: offset_x + patch_w + 2 * pad] += \
                seg_pred
            edge_out[:, :, offset_y:offset_y + patch_h + 2 * pad, offset_x: offset_x + patch_w + 2 * pad] += \
                edge_pred
            mat_count[:, :, offset_y:offset_y + patch_h + 2 * pad, offset_x: offset_x + patch_w + 2 * pad] += 1.0

    sed_out = torch.divide(sed_out, mat_count)
    seg_out = torch.divide(seg_out, mat_count)
    edge_out = torch.divide(edge_out, mat_count)

    # crop padding
    if pad != 0:
        sed_out = sed_out[:, :, pad: -pad, pad: -pad]
        seg_out = seg_out[:, :, pad: -pad, pad: -pad]
        edge_out = edge_out[:, :, pad: -pad, pad: -pad]

    return sed_out, seg_out, edge_out


# save orig edge masks
def save_sed_pred_masks(sed_pred, img_info, n_classes, factor, out_folder):
    '''
    :param sed_pred:  np.ndarray
    :param img_info: dict, impath, gtpath, orig_size
    :param n_classes: int
    :param out_folder: str, output folder
    '''
    height, width = img_info["orig_size"]
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    for i in range(n_classes):
        temp_mask = sed_pred[i, :height, :width]
        temp_mask = temp_mask * 255 * factor
        temp_mask = np.where(temp_mask > 255, 255, temp_mask)
        im = temp_mask.astype(np.uint8)
        out_path = os.path.join(out_folder, f"class_{i + 1}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path, filename), im)


# save edge masks higher than thresh
def save_sed_binary_masks(sed_pred, img_info, n_classes, thresh, out_folder):
    '''
    :param sed_pred:  np.ndarray
    :param img_info: dict, impath, gtpath, orig_size
    :param n_classes: int
    :param thresh len(thresh) = n_classes, thresh \in (0, 1)
    :param out_folder: str, output folder
    '''
    height, width = img_info["orig_size"]
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    for i in range(n_classes):
        temp_mask = sed_pred[i, 0:height, 0:width]
        temp_mask[temp_mask >= 255] = 0
        temp_mask = np.where(temp_mask > thresh[i], 255, 0)
        im = temp_mask.astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        out_path = os.path.join(out_folder, f"class_{i + 1}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        cv2.imwrite(os.path.join(out_path, filename), im)


# save visual sed result
def save_visual_sed_pred(sed_pred, n_classes, img_info, dataset, thresh, out_folder):
    '''
    :param sed_pred:  np.ndarray, cls*H*W
    :param n_classes: classs number
    :param img_info: dict, impath, gtpath, orig_size
    :param dataset: choice=[cityscapes, sbd]
    :param thresh: list,len(thresh)=n_classes
    :param out_folder: output folder
    :return: None
    '''

    height, width = img_info["orig_size"]
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    visual_img = np.zeros((height, width, 3))
    edge_sum = np.zeros((height, width))
    colors = get_visual_colors(dataset)

    for i in range(n_classes):
        temp_mask = sed_pred[i, 0:height, 0:width]
        temp_mask[temp_mask >= 255] = 0
        temp_mask = np.where(temp_mask >= thresh[i], 1, 0)
        edge_sum += temp_mask
        for c in range(3):
            visual_img[:, :, c] = np.where(temp_mask == 1, visual_img[:, :, c] + colors[i][c], visual_img[:, :, c])
    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    visual_img[idx] = visual_img[idx] / edge_sum[idx]
    visual_img[~idx] = 255

    out_file = os.path.join(out_folder, filename)
    visual_img = cv2.cvtColor(visual_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_file, visual_img)


# save seg result
def save_seg_pred(seg_pred, img_info, out_folder):
    '''
    :param seg_pred: np.ndarray, cls*H*W
    :param img_info: dict, impath, gtpath, orig_size
    :param out_folder: output folder
    :return: None
    '''
    seg_result = np.argmax(seg_pred, axis=0).astype(np.uint8)
    height, width = img_info["orig_size"]
    seg_result = cv2.resize(seg_result, (width, height), interpolation=cv2.INTER_NEAREST)
    seg_result = cv2.cvtColor(seg_result, cv2.COLOR_GRAY2BGR)
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    cv2.imwrite(os.path.join(out_folder, filename), seg_result)


# save visual seg result
def save_visual_seg_pred(seg_pred, dataset, img_info, out_folder):
    '''
    :param seg_pred:  np.ndarray, cls*H*W
    :param dataset: str, choice=["cityscapes", "sbd"]
    :param img_info: dict, impath, gtpath, orig_size
    :param out_folder: output folder
    :return: None
    '''
    seg_result = np.argmax(seg_pred, axis=0)
    colors = get_visual_colors(dataset)
    height, width = img_info["orig_size"]
    seg_result = cv2.resize(seg_result, (width, height), interpolation=cv2.INTER_NEAREST)
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    visual_seg = np.zeros((height, width, 3))
    for row in range(height):
        for col in range(width):
            visual_seg[row, col, :] = np.array(colors[int(seg_result[row, col])])

    cv2.imwrite(os.path.join(out_folder, filename), visual_seg)


# save edge pred
def save_edge_pred(edge_pred, img_info, out_folder):
    '''
    :param edge_pred: np.ndarray, H*W
    :param img_info: dict, impath, gtpath, orig_size
    :param out_folder: output folder
    :return: None
    '''
    edge_pred = (edge_pred * 255).astype(np.uint8)
    height, width = img_info["orig_size"]
    edge_pred = cv2.resize(edge_pred, (width, height))
    filename = os.path.basename(img_info["impath"]).split(".")[0] + ".png"
    edge_pred = cv2.cvtColor(edge_pred, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(os.path.join(out_folder, filename), edge_pred)


# def save_visual_feature(feature, )

def main(args):
    # print args
    print("****")
    print(args)
    print("****")

    # dataset for inference
    dataset_ = DataList(args.data_root, args.file_list, args.n_classes)
    file_num = len(dataset_)

    # init models
    model_list = nn.ModuleList()

    # init models
    # todo: custom model
    from models.backbone import ResNet101
    from save.city.v06.ICE_MHACA_BCA import ACA
    backbone = ResNet101()
    device_id = [0, ]
    backbone = torch.nn.DataParallel(backbone.cuda(), device_ids=device_id)

    net = ACA(nclasses=19, inter_channel=256, rank_=256)
    device_id = [0, ]
    net = torch.nn.DataParallel(net.cuda(), device_ids=device_id)

    # load ckpt
    ckpt = torch.load(args.ckpt)
    backbone.load_state_dict(ckpt["extractor"], strict=True)
    net.load_state_dict(ckpt["model"], strict=True)

    backbone.eval()
    net.eval()

    model_list.append(backbone)
    model_list.append(net)

    # load thresh
    if args.thresh is not None:
        thresh = load_thresh(args.thresh, args.n_classes)
        # print(thresh)
    else:
        thresh = [0.2] * args.n_classes

    for i in range(len(thresh)):
        thresh[i] = thresh[i] if thresh[i] > 0.2 else 0.2


    # predict
    for idx in tqdm.tqdm(range(file_num)):
        img, gt, seg, edge, img_info = dataset_[idx]
        # img: BGR - mean_value, torch.tensor, 3*H*W
        # gt: sed_bin_mask, torch.tensor, cls*H*W
        # seg: seg_mask, torch.tensor, 1*H*W
        # edge: edge_bin_mask, torch.tensor, 1*H*W
        # image_info: (height , width)

        # convert format if needed

        # predict
        if not args.use_patch_pred:
            # for full image predict, no sigmoid !!!
            sed_pred, seg_pred, edge_pred = image_predict(model_list, img.cuda())
        else:
            # for patch predict no sigmoid !!!
            sed_pred, seg_pred, edge_pred = patch_predict(model_list, img.cuda(), n_classes=args.n_classes,
                                                          patch_h=640, patch_w=640, step_size_x=352, step_size_y=384,
                                                          pad=16)

        # save predict
        # for sed_
        sed_pred = torch.squeeze(sed_pred, 0).cpu().numpy()
        # save sed pred masks
        out_folder = os.path.join(args.out_dir, "classes")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        save_sed_pred_masks(sed_pred, img_info, args.n_classes, args.factor, out_folder)

        # # save sed binary masks
        out_folder = os.path.join(args.out_dir, "bin_sed_masks")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        save_sed_binary_masks(sed_pred, img_info, args.n_classes, thresh, out_folder)

        # # save visual sed predict
        out_folder = os.path.join(args.out_dir, "visual_sed_pred")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        save_visual_sed_pred(sed_pred, args.n_classes, img_info, args.dataset, thresh, out_folder)

        # for seg
        seg_pred = torch.squeeze(seg_pred, 0).cpu().numpy()
        # save seg pred
        out_folder = os.path.join(args.out_dir, "seg_pred")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        save_seg_pred(sed_pred, img_info, out_folder)
        # save visual seg pred
        out_folder = os.path.join(args.out_dir, "visual_seg_pred")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        save_visual_seg_pred(seg_pred, args.dataset, img_info, out_folder)

        # for edge
        out_folder = os.path.join(args.out_dir, "visual_edge_pred")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        edge_pred = torch.squeeze(edge_pred).cpu().numpy()
        save_edge_pred(edge_pred, img_info, out_folder)

        # for feature visualization
        # save_visual_feature()

        # visual imgs and gt
        # from imlist
        # imlist = default_flist_reader(args.file_list)


if __name__ == '__main__':
    # parse arg
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="project root")
    # dataset
    parser.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "sbd"],
                        help="specify dataset name")
    parser.add_argument("--data-root", type=str, default="dataset_name/data_proc", help="gtFine and leftImg8bit in dir")
    parser.add_argument("--file-list", type=str, default="dataset_name/data_proc/val.txt",
                        help="Line format: image_path edge_bin_path")
    parser.add_argument("--n_classes", type=int, help="number of classes")
    # ckpt
    parser.add_argument("--ckpt", type=str, help="ckpt file path")
    # thresh or factor
    parser.add_argument("--factor", type=float, default=1.0, help="factor on results")
    parser.add_argument("--thresh", type=str, default=None, help=".mat result files in the dir")
    # out_folder
    parser.add_argument("--out-dir", type=str, default=None, required=True, help="root dir of outputs")
    # use_patch_pred
    parser.add_argument("--use-patch-pred", type=bool, default=True)
    args = parser.parse_args()

    # change root dir
    if args.root is not None:
        os.chdir(args.root)
        sys.path.append(args.root)

    main(args)

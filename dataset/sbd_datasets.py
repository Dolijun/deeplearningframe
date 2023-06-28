# coding:utf-8
from torch.utils.data import Dataset
import cv2
import random
import os
import os.path
import numpy as np
import torch
import scipy.io as scio


class DataList(Dataset):
    def __init__(self, root, flist, n_classes):
        self.root = root
        self.imlist = self.default_flist_reader(flist)
        self.n_classes = n_classes
        self.mean_value = (104.008, 116.669, 122.675)  # BGR

    # 1.读取文件索引列表 .txt格式(img,label)
    def default_flist_reader(self, flist):
        """
        flist format: impath label\nimpath label\n ...(same to caffe's filelist)
        """
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                # if '/scale_1/' in line:
                #     splitted = line.strip().split()
                #     if len(splitted) == 2:
                #         impath, imlabel = splitted
                #     elif len(splitted) == 1:
                #         impath, imlabel = splitted[0], None
                #     else:
                #         raise ValueError('weird length ?')
                #     impath = impath.strip('../')
                #     imlist.append((impath, imlabel))
                splitted = line.strip().split()
                if len(splitted) == 2:
                    impath, imlabel = splitted
                elif len(splitted) == 1:
                    impath, imlabel = splitted[0], None
                else:
                    raise ValueError('weird length ?')
                impath = impath.strip('../')
                imseg = impath.replace('image', 'gt_catagory')
                imseg = imseg.replace('png', 'mat')
                imlist.append((impath, imlabel, imseg))
        return imlist

    # 2.把位真值转换成通道真值
    def binary_file_to_channel_masks(self, bin_file, h, w, channels, ignore_pixel_id_map=(31, 255)):
        array = np.fromfile(bin_file, dtype=np.uint32)
        array = array.reshape(h, w)
        arr_chn = np.zeros((channels, h, w), dtype=np.float32)
        ignore_mask = array & (1 << ignore_pixel_id_map[0]) > 0
        for c in range(channels):
            mask = array & (1 << c) > 0
            arr_chn[c, :, :] = mask
            arr_chn[c, :, :][ignore_mask] = ignore_pixel_id_map[1]
        return arr_chn.transpose((1, 2, 0))

    # 4.图片预处理
    def my_transform(self, in_, gt_mask, seg):
        return in_, gt_mask, seg

    # 5.裁剪
    def auto_crop(self, crop_size, input, target, seg, is_fix=False):
        img_width, img_height = input.shape[1], input.shape[0]
        pad_height = max(crop_size - img_height, 0)
        pad_width = max(crop_size - img_width, 0)
        if pad_height > 0 or pad_width > 0:
            input = cv2.copyMakeBorder(input, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)
            target = cv2.copyMakeBorder(target, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[255] * 4)
            seg = cv2.copyMakeBorder(seg, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=255)
        width, height = input.shape[1], input.shape[0]
        transX = random.randint(0, width - crop_size)
        transY = random.randint(0, height - crop_size)
        if is_fix:
            transX = 0
            transY = 0
        input = input[transY:transY + crop_size, transX:transX + crop_size, :]
        target = target[transY:transY + crop_size, transX:transX + crop_size, :]
        seg = seg[transY:transY + crop_size, transX:transX + crop_size]
        return np.array(input), np.array(target), np.array(seg)

    def __getitem__(self, index):
        impath, gtpath, imseg = (
            os.path.join(self.root, *self.imlist[index][0].split('/')),
            os.path.join(self.root, *self.imlist[index][1].split('/')),
            os.path.join(self.root, *self.imlist[index][2].split('/'))
        )
        image = cv2.imread(impath).astype(np.float32)
        width, height = image.shape[1], image.shape[0]
        image -= np.array(self.mean_value)
        gt = self.binary_file_to_channel_masks(gtpath, height, width, self.n_classes)
        seg_mat = scio.loadmat(imseg)
        if 'mask' in seg_mat.keys():
            seg_mask = seg_mat['mask']
        elif 'segCls' in seg_mat.keys():
            seg_mask = seg_mat['segCls']
        seg_mask[seg_mask == 0] = self.n_classes + 1
        seg_mask = seg_mask - 1  # 0-19 class 20 background
        image, gt, seg = self.my_transform(image, gt, seg_mask)
        image = image.transpose((2, 0, 1))  # HxWx3 -> 3xHxW
        gt = gt.transpose((2, 0, 1))
        # seg = cv2.resize(seg, (seg.shape[0] / 2, seg.shape[1] / 2), interpolation=cv2.INTER_NEAREST)
        edge = np.max(gt, axis=0)
        # edge = cv2.resize(edge, (edge.shape[0] / 2, edge.shape[1] / 2), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image)
        gt = torch.from_numpy(gt)
        seg = torch.from_numpy(seg)
        edge = torch.from_numpy(edge)
        edge = edge.unsqueeze(0)
        image_info = {'impath': impath, 'gtpath': gtpath, 'orig_size': (height, width)}
        return image, gt, seg, edge, image_info

    def __len__(self):
        return len(self.imlist)


class SBDTrain(DataList):
    def __init__(self, root, flist, crop_size=352):
        super(SBDTrain, self).__init__(root, flist, n_classes=20)
        self.crop_size = crop_size

    def my_transform(self, input, target, seg):
        # Random hflip
        hflip = random.random()
        if (hflip < 0.5):
            input = np.fliplr(input)
            target = np.fliplr(target)
            seg = np.fliplr(seg)
        # Random crop
        input, target, seg = self.auto_crop(self.crop_size, input, target, seg)
        return input, target, seg


class SBDVal(DataList):
    def __init__(self, root, flist, crop_size=352):
        super(SBDVal, self).__init__(root, flist, n_classes=20)
        self.crop_size = crop_size

    def my_transform(self, input, target, seg):
        # Crop
        input, target, seg = self.auto_crop(self.crop_size, input, target, seg, is_fix=True)
        return input, target, seg

    def train_data(self, index):
        input, target, seg, edge, _ = self.__getitem__(index)
        input, target, seg, edge = input.unsqueeze(dim=0), target.unsqueeze(dim=0), seg.unsqueeze(dim=0), edge.unsqueeze(dim=0)  # 1x3xHxW
        return input, target, seg, edge


class SBDTest(DataList):
    def __init__(self, root, flist, crop_size=512):
        super(SBDTest, self).__init__(root, flist, n_classes=20)
        self.crop_size = crop_size

    def my_transform(self, input, target, seg):
        # Crop
        input, target, seg = self.auto_crop(self.crop_size, input, target, seg)
        return input, target, seg
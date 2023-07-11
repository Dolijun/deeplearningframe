import os
import cv2
import numpy as np
import json


def labelme2mask_single_img(img_path, label_me_json_path, label_info):
    # 读取图像
    img = cv2.imread(img_path)

    # 0-背景 空白图像
    img_mask = np.zeros(img.shape[:2])

    # 加载json文件
    with open(label_me_json_path, "r", encoding='utf-8') as f:
        labelme = json.load(f)
    # 遍历所有类别
    for object_ in labelme:
        label = object_['label']
        shape = object_['points']
        type = object_['shape_type']

        if type == "polygon":
            shape = [np.array(shape, dtype=np.int32).reshape((-1, 1, 2))]

            # 画 mask
            img_mask = cv2.fillPoly(img_mask, shape, color=label_info[label]['color'])

        elif type == "line":
            shape = [np.array(shape, dtype=np.int32).reshape((-1, 1, 2))]

            # 画 edge
            img_mask = cv2.polylines(img_mask, shape, isClose=False, color=label_info[label]['color'],
                                     thickness=label_info[label]['thickness'])
    return img_mask


if __name__ == '__main__':
    class_info = {'Object': {'type': 'polygon', 'color': 255, 'thickness': 3}}
    root = f"./temp_file/temp"
    files = os.listdir(root)
    filenames = []
    for file in files:
        if file.endswith(".json"):
            filenames.append(file[:-5])

    for filename in filenames:
        img_path = os.path.join(root, filename + ".png")
        json_path = os.path.join(root, filename + ".json")
        mask_path = os.path.join(root, filename + "_mask.png")

        img_mask = labelme2mask_single_img(img_path, json_path, class_info)

        cv2.imwrite(mask_path, img_mask)
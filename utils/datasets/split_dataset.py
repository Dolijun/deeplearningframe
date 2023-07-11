import os
import shutil
import random
import pandas as pd


def split_dataset(data_path, val_frac=0.2, seed_=123):
    '''
    directory struction:
    dataset01
        class01
        class02
            01.png
            02.png
            ...
        ...
    dataset02
    ...
    '''
    classes = os.listdir(data_path)
    # print(classes)

    # makedir, train and val
    if not os.path.exists(os.path.join(data_path, "train")):
        os.makedirs(os.path.join(data_path, "train"))
    if not os.path.exists(os.path.join(data_path, "val")):
        os.makedirs(os.path.join(data_path, "val"))

    for class_ in classes:
        if not os.path.exists(os.path.join(data_path, "train", class_)):
            os.makedirs(os.path.join(data_path, "train", class_))
        if not os.path.exists(os.path.join(data_path, "val", class_)):
            os.makedirs(os.path.join(data_path, "val", class_))

    # df = pd.DataFrame()
    print('{:^18}{:^18}{:^18}'.format("class_", "train_pic_num", "val_pic_num"))
    for class_ in classes:
        img_files = os.listdir(os.path.join(data_path, class_))

        # shuffle
        random.shuffle(img_files)

        # split
        val_num = int(len(img_files) * val_frac)
        val_imgs = img_files[:val_num]
        train_imgs = img_files[val_num:]

        # move imgs
        for img in val_imgs:
            old_img_path = os.path.join(data_path, class_, img)
            new_img_path = os.path.join(data_path, 'val', class_, img)
            shutil.move(old_img_path, new_img_path)

        for img in train_imgs:
            old_img_path = os.path.join(data_path, class_, img)
            new_img_path = os.path.join(data_path, 'train', class_, img)
            shutil.move(old_img_path, new_img_path)

        assert len(os.listdir(os.path.join(data_path, class_))) == 0
        shutil.rmtree(os.path.join(data_path, class_))
        train_num = len(os.listdir(os.path.join(data_path, 'train', class_)))
        val_num = len(os.listdir(os.path.join(data_path, 'val', class_)))
        print('{:^18}{:^18}{:^18}'.format(class_, train_num, val_num))
        # df.append({'class':class_, 'train_num':train_num, 'val_num':val_num})

    # df['total'] = df['train_num'] + df['val_num']
    # df.to_csv('dataset_statics.csv', index=False)

if __name__ == "__main__":
    split_dataset('/home/ilab/dolijunc/project/mmpretrain/data/fruit30')
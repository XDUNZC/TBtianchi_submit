# -*- coding: utf-8 -*-
'''
@time: 2019/01/11 11:28
spytensor
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
from IPython import embed
from sklearn.model_selection import train_test_split

np.random.seed(41)

# 0为背景
classname_to_id = {'长马甲': 1, '古风': 2, '短马甲': 3, '背心上衣': 4, '背带裤': 5, '连体衣': 6, '吊带上衣': 7, '中裤': 8, '短袖衬衫': 9,
                   '无袖上衣': 10,
                   '长袖衬衫': 11, '中等半身裙': 12, '长半身裙': 13, '长外套': 14, '短裙': 15, '无袖连衣裙': 16, '短裤': 17, '短外套': 18,
                   '长袖连衣裙': 19, '长袖上衣': 20, '长裤': 21, '短袖连衣裙': 22, '短袖上衣': 23, '古装': 2}


class Csv2CoCo:

    def __init__(self, image_dir, total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi, label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            if k == '古装':
                continue
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        #         print(path)
        img = cv2.imread(self.image_dir + path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path.split(os.sep)[-2] + '_' + path.split(os.sep)[-1]
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    # 计算面积
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x + 1) * (max_y - min_y + 1)

    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x, min_y, min_x, min_y + 0.5 * h, min_x, max_y, min_x + 0.5 * w, max_y, max_x, max_y, max_x,
                  max_y - 0.5 * h, max_x, min_y, max_x - 0.5 * w, min_y])
        return a


def creat_csv_image_dataset(data_root_path='origin_data/', mode='train'):
    dataset_paths = glob.glob(data_root_path + mode + '*')
    # 图像库中标注
    img_ann_folder_paths = []  # 所有data/train_dataset_part<n>/image_annotatonl中所有文件夹

    #     # 视频库中标注
    #     video_ann_paths = []  # 所有data/train_dataset_part<n>/video_annotation中所有json文件

    for dataset_path in dataset_paths:
        img_ann_folder_paths.extend(glob.glob(dataset_path + '/image_annotation/*'))

    #         video_ann_paths.extend(glob.glob(dataset_path + '/video_annotation/*.json'))

    image_db = []
    for img_ann_folder_path in img_ann_folder_paths[:]:
        split_list = img_ann_folder_path.split('/')
        img_folder_path = data_root_path + split_list[1] + '/image/' + split_list[-1] + '/'
        json_paths = glob.glob(img_ann_folder_path + '/*.json')
        for json_path in json_paths:
            with open(json_path, 'r') as f:
                img_anns = json.load(f)
            if len(img_anns['annotations']) > 0:
                img_path = img_folder_path + json_path.split('/')[-1].split('.')[0] + '.jpg'
                for ann in img_anns['annotations']:
                    box = ann['box']
                    label = ann['label']
                    image_db.append([img_path, box[0], box[1], box[2], box[3], label])

    image_db = pd.DataFrame(image_db)

    image_db.to_csv(data_root_path + mode + '_image_dataset.csv', index=False)
    print('已生成csv路径文件：' + data_root_path + mode + '_image_dataset.csv')
    print(image_db.info())

def creat_coco_train_val():
    csv_file = "origin_data/train_image_dataset.csv"
    image_dir = ""
    saved_coco_path = ""
    # 整合csv格式标注文件
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file).values
    # num=0
    for annotation in annotations:  # origin_data/validation_dataset_part1/image/064598/0.jpg,233,156,530,755,短袖连衣裙
        #     key = annotation[0].split(os.sep)[-2]+'_'+annotation[0].split(os.sep)[-1]
        key = annotation[0]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value
    #     num+=1
    #     if num==10:
    #         break
    # 按照键值划分数据
    total_keys = list(total_csv_annotations.keys())
    train_keys, val_keys = train_test_split(total_keys, test_size=0.2)
    print("train_n:", len(train_keys), 'val_n:', len(val_keys))
    # 创建必须的文件夹
    if not os.path.exists('%scoco/annotations/' % saved_coco_path):
        os.makedirs('%scoco/annotations/' % saved_coco_path)
    if not os.path.exists('%scoco/train2017/' % saved_coco_path):
        os.makedirs('%scoco/train2017/' % saved_coco_path)
    if not os.path.exists('%scoco/val2017/' % saved_coco_path):
        os.makedirs('%scoco/val2017/' % saved_coco_path)
    # 把训练集转化为COCO的json格式
    l2c_train = Csv2CoCo(image_dir=image_dir, total_annos=total_csv_annotations)
    train_instance = l2c_train.to_coco(train_keys)
    l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json' % saved_coco_path)
    for file in train_keys:
        if not os.path.exists("%scoco/train2017/%s" % (
                saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1])):
            print(
                "%scoco/train2017/%s" % (saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1]))
            shutil.copy(image_dir + file, "%scoco/train2017/%s" % (
                saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1]))
    for file in val_keys:
        if not os.path.exists(
                "%scoco/val2017/%s" % (saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1])):
            shutil.copy(image_dir + file, "%scoco/val2017/%s" % (
                saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1]))
    # 把验证集转化为COCO的json格式
    l2c_val = Csv2CoCo(image_dir=image_dir, total_annos=total_csv_annotations)
    val_instance = l2c_val.to_coco(val_keys)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json' % saved_coco_path)

def crear_coco_test():
    csv_file = "origin_data/test_image_dataset.csv"
    image_dir = ""
    saved_coco_path = ""
    # 整合csv格式标注文件
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file).values
    # num=0
    for annotation in annotations:  # origin_data/validation_dataset_part1/image/064598/0.jpg,233,156,530,755,短袖连衣裙
        #     key = annotation[0].split(os.sep)[-2]+'_'+annotation[0].split(os.sep)[-1]
        key = annotation[0]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value
    #     num+=1
    #     if num==10:
    #         break
    # 按照键值划分数据
    test_keys = list(total_csv_annotations.keys())
    print("test_n:", len(test_keys))
    # 创建必须的文件夹
    if not os.path.exists('%scoco/annotations/' % saved_coco_path):
        os.makedirs('%scoco/annotations/' % saved_coco_path)
    if not os.path.exists('%scoco/train2017/' % saved_coco_path):
        os.makedirs('%scoco/train2017/' % saved_coco_path)
    if not os.path.exists('%scoco/val2017/' % saved_coco_path):
        os.makedirs('%scoco/val2017/' % saved_coco_path)
    if not os.path.exists('%scoco/itest2017/' % saved_coco_path):
        os.makedirs('%scoco/test2017/' % saved_coco_path)
    # 把训练集转化为COCO的json格式
    l2c_test = Csv2CoCo(image_dir=image_dir, total_annos=total_csv_annotations)
    test_instance = l2c_test.to_coco(test_keys)
    l2c_test.save_coco_json(test_instance, '%scoco/annotations/instances_test2017.json' % saved_coco_path)
    for file in test_keys:
        if not os.path.exists(
                "%scoco/test2017/%s" % (saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1])):
            print(
                "%scoco/test2017/%s" % (saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1]))
            shutil.copy(image_dir + file, "%scoco/test2017/%s" % (
            saved_coco_path, file.split(os.sep)[-2] + '_' + file.split(os.sep)[-1]))


if __name__ == '__main__':
    # 从原始数据生成csv
    creat_csv_image_dataset()
    creat_csv_image_dataset(mode='test')
    # 从csv转到coco
    creat_coco_train_val()
    crear_coco_test()



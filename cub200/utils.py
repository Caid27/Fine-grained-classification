import os
import string
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt




def read_data(root:str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    cub_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    cub_class.sort()
    cub_class = cub_class[1:] #在平台上要添加，因为会默认添加ipynb_checkpoints文件
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k,name.split(".")[-1]) for k,name in enumerate(cub_class))
    json_str = json.dumps(dict((key,test) for key, test in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    with open("train_test_split.txt") as file:
        train_test_split = dict((num,line.split(" ")[-1])for num,line in enumerate(file))
        
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = []  # 存储测试集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    pic_num = 0
    for num,cla in enumerate(cub_class):
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取当前文件夹所代表的类
        image_class = class_indices[num]
        
        for img_path in images:
            a = train_test_split[pic_num]
            if train_test_split[pic_num]=='1\n':  # 如果是测试集
                test_images_path.append(img_path)
                test_images_label.append(num)
            else:                             
                train_images_path.append(img_path)
                train_images_label.append(num)
            pic_num+=1


    print("{} images for training.".format(len(train_images_path)))
    print("{} images for test.".format(len(test_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(test_images_path) > 0, "number of testidation images must greater than 0."

    return train_images_path, train_images_label, test_images_path, test_images_label



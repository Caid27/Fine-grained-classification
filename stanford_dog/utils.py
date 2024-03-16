import os
import string
import sys
import json
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io as sio



def read_mat(root:str,path:str):
    assert os.path.exists(path), "dataset root: {} does not exist.".format(path)
    # 加载MATLAB的.mat文件
    mat_contents = sio.loadmat(path)
    # 提取label和file_list数据
    train_test_labels = mat_contents['labels'] # 12000*1的double类型数组
    file_list = mat_contents['file_list'] # 12000*1的cell类型
    # 转置
    train_test_labels = train_test_labels.transpose()[0].tolist()
    file_list = file_list.transpose()[0].tolist()
    # 补全路径
    file_list = [os.path.join(root,i[0]) for i in file_list]
    # label从0开始
    train_test_labels = [i-1 for i in  train_test_labels]
    return train_test_labels,file_list


def read_data(root:str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    dog_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    dog_class.sort()
    dog_class = dog_class[1:] #在平台上要添加，因为会默认添加ipynb_checkpoints文件
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k,name.split("-")[-1]) for k,name in enumerate(dog_class))
    json_str = json.dumps(dict((key,test) for key, test in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_images_path = []  # 存储测试集的所有图片路径
    test_images_label = []  # 存储测试集图片对应索引信息

    train_images_label,train_images_path = read_mat(root,"./train_list.mat")

    test_images_label, test_images_path = read_mat(root,"./test_list.mat")

    print("{} images for training.".format(len(train_images_path)))
    print("{} images for test.".format(len(test_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(test_images_path) > 0, "number of testidation images must greater than 0."

    return train_images_path, train_images_label, test_images_path, test_images_label




if __name__ =="__main__":
    train_images_path, train_images_label, test_images_path, test_images_label = read_data("images")





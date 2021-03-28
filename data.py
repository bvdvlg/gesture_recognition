from torch.utils.data import Dataset, Subset
import numpy as np
from progress.bar import IncrementalBar
import os
import image_container
from image_container import dots_path
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetHands(Dataset):
    def __init__(self, path=None, transform_data=None, transform_labels=None, data=None, labels=None):
        if labels is None:
            labels = list()
        if data is None:
            data = list()

        self.data, self.labels = data, labels

        self.unique_labels = None
        self.reversed_labels = None

        self.transform_data = transform_data
        self.transform_labels = transform_labels

        if path is not None:
            self.load(path)

        self.unique_labels = {el: num for num, el in enumerate(sorted(list(set(self.labels))))}
        self.reversed_labels = {num: el for num, el in enumerate(sorted(list(set(self.labels))))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data[index]
        label = self.labels[index]

        if self.transform_data is not None:
            image = self.transform_data(image)
        if self.transform_labels is not None:
            label = self.transform_labels(label, self.unique_labels)

        return image, label

    @staticmethod
    def collect(source):
        images = {}
        for suffix in os.listdir(source):
            bar = IncrementalBar(suffix, max=len(os.listdir("{}/{}".format(source, suffix))))
            images[suffix] = list()
            for file in os.listdir("{}/{}".format(source, suffix)):
                image = image_container.Image.fread("{}/{}/{}".format(source, suffix, file))
                images[suffix].append(image)
                bar.next()
            bar.finish()
        return images

    def load(self, path=dots_path):
        data = self.collect(path)

        bar = IncrementalBar("Creating dataset", max=sum([len(arr) for arr in data.values()]))
        for array in data.values():
            for value in array:
                data, label = value.get_data()
                self.data.append(data)
                self.labels.append(label)
                bar.next()
        bar.finish()

    def train_test_split(self, ratio):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=ratio)
        train_data = DatasetHands(data=X_train, labels=y_train,
                                  transform_data=self.transform_data, transform_labels=self.transform_labels)
        test_data = DatasetHands(data=X_test, labels=y_test,
                                 transform_data=self.transform_data, transform_labels=self.transform_labels)
        return train_data, test_data


def torch_data_transformer(data):
    return torch.FloatTensor(data)


def torch_label_transformer(label, dic):
    return dic[label]


def xgb_data_transformer(data):
    data = [el for el in data]
    help = data[:3]
    for i in range(len(data)):
        data[i] -= help[i % 3]
        data[i] *= 10
    data.append(data[36]-data[24])
    data.append(data[24]-data[12])
    data.append(data[25]-data[19])
    data.append(data[30]-data[12])
    data.append(data[42]-data[12])
    data.append(data[43]-data[13])
    data.append(data[54]-data[13])
    dist = (data[15]**2+data[16]**2+data[17]**2)**(1/2)
    data = [el/dist for el in data]
    return pd.Series(data)


def xgb_label_transformer(label, dic):
    return dic[label]

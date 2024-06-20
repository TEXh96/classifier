#coding utf-8
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from collections import Counter

try:
    from .augmentation import augmentaion_factory
except:
    from augmentation import augmentaion_factory #调试当前脚本

aug = augmentaion_factory['pantograph']


class PantographDataset(Dataset):
    """受电弓分类数据集定义"""

    def __init__(self, annotations_txt, is_train=False):
        try:
            with open(annotations_txt, 'r') as f:
                self.annotations = f.readlines()
                self._cal_probability()
        except Exception as e:
            print("error: {}".format(e))

        self.is_train = is_train
        self.aug = None
        if self.is_train:
            self.aug = aug()
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023)), #bgr order
            ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        while True:
            prob = np.random.random()
            annotation = self.annotations[idx]
            image_file, label = annotation.rstrip().split(' ')
            if prob < self.probability[label]:
                break
            else:
                idx = (idx+1)%len(self.annotations)
        # print(image_file)
        label = int(label)
        image_bgr = cv2.imread(image_file)
        if self.aug:
            image_bgr = self.aug(image=image_bgr)['image']
        PIL_image = Image.fromarray(image_bgr)
        imgdata = self.transform(PIL_image)

        return [imgdata, label]

    def _cal_probability(self):
        labels = [annotation.rstrip().split(' ')[-1] for annotation in self.annotations]
        frequency = dict(Counter(labels))
        print("result: ", frequency)
        self.probability = {key: value/len(labels) for key, value in frequency.items()}
        min_probability = min([value for key, value in self.probability.items()])
        self.probability = {key: min_probability/value for key, value in self.probability.items()}
        print("probability: ", self.probability)

    def get_num_classes(self):
        return len(self.probability)


if __name__ == '__main__':
    annotations_txt = '/data2/wangj/research/image_classification/classifier/datasets/sdg_blur/annotations_train.txt'
    trainset = PantographDataset(annotations_txt, is_train=True)

    for idx, (image, label) in enumerate(trainset, 1):
        image = image.permute(1, 2, 0).numpy() # CHW to HWC
        image = image * np.array([51.255, 50.847, 51.5865])
        image = image + np.array([113.8575, 122.961, 125.307]) 
        image = image.astype('uint8')
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        print("label: ", label)
        cv2.imwrite('pantograph.jpg', image)
        input()

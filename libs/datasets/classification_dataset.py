#coding: utf-8

'''
##图像分类任务
    数据集制作流程备忘：
        1. 原始数据抽帧
        2. 数据标注得到对应的json文件，并按加样日期建立文件夹存放到固定目录下
        3. 读取json文件信息，按labelname对原图做裁剪分类整理（裁剪时做适当外扩，便于后绪做数据增强）
        4. 脚本生成各日期对应的train_xxx.txt, val_xxx.txt, 并合成为tarin.txt, val.txt, 以上txt按日期创建到具体任务的子目录下：
            taskname_folder:
                日期_folder:
                    train_xxx.txt
                    val_xxx.txt
                    ...
                    train.txt #训练集文件
                    val.txt #验证集文件
                    label_map.json #txt中的labelname和label对应关系

        注意：2.3步的流程看具体问题确定，重要的是要在原图上标注(可能只需要按类别分下文件夹)
    
    数据集加载流程备忘：
        直接参考当前脚本文件
'''

from torch.utils.data import Dataset
import json
import cv2
import numpy as np



# 图像缩放: 保持图片的宽高比例，剩下的部分采用灰色填充。
def letterbox(img, new_shape=(224, 224), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) # 计算缩放因子
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])#(640/720,640/1280)
    """
    缩放(resize)到输入大小img_size的时候，如果没有设置上采样的话，则只进行下采样
    因为上采样图片会让图片模糊，对训练不友好影响性能。
    """
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)#

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) #(240,640) W,H
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle # 获取最小的矩形填充
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding 倍数的余数
    # 如果scaleFill=True,则不进行填充，直接resize成img_size, 任由图片进行拉伸和压缩
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    # 计算上下左右填充大小
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 进行填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)






class ClassificationDataset(Dataset):
    
    def __init__(
        self,
        annotations_txt, 
        label_map_file,      
        augmentation=None,
        preprocessing=None 
    ):
        try:
            with open(annotations_txt, 'r') as f:
                self.annotations = f.readlines()
        except Exception as e:
            print("error: {}".format(e))  
        
        try:
            with open(label_map_file, "r") as f:
                self.label_map = json.load(f)
        except Exception as e:
            print("error: {}".format(e)) 

        # filter annotations,可能只需要训练一部分类别
        self.annotations = [annotation for annotation in self.annotations \
            if annotation.rstrip().split(' ')[-1] in self.label_map.keys()]

        self.augmentation = augmentation
        self.preprocessing = preprocessing      

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        #print("annotation.rstrip().split(' ')[0]: ",annotation.rstrip().split(' ')[0])
        #print("annotation.rstrip().split(' ')[1]: ",annotation.rstrip().split(' ')[1])
        #if len(annotation.rstrip().split(' ')==3):
            #image_file_0,image_file_1, labelname = annotation.rstrip().split(' ')
            #image_file = image_file_0+image_file_1
        #else:
            #image_file, labelname = annotation.rstrip().split(' ')
        image_file, labelname = annotation.rstrip().split(' ')   
        image_src = cv2.imread(image_file)
        image = letterbox(image_src)[0]
        print("channnel: ",image.channnel())
        cv2.imwrite("./lettebox.jpg",image)
        
        # if 1:
        #     print("image_file: ", image_file)
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # note: Albumentation use rgb image format

        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]
        
        return image, int(self.label_map[labelname])
        
        
class ClassificationDataset_1280_1280(Dataset):
    
    def __init__(
        self,
        annotations_txt, 
        label_map_file,      
        augmentation=None,
        preprocessing=None 
    ):
        try:
            with open(annotations_txt, 'r') as f:
                self.annotations = f.readlines()
        except Exception as e:
            print("error: {}".format(e))  
        
        try:
            with open(label_map_file, "r") as f:
                self.label_map = json.load(f)
        except Exception as e:
            print("error: {}".format(e)) 

        # filter annotations,可能只需要训练一部分类别
        self.annotations = [annotation for annotation in self.annotations \
            if annotation.rstrip().split(' ')[-1] in self.label_map.keys()]

        self.augmentation = augmentation
        self.preprocessing = preprocessing      

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        #print("annotation.rstrip().split(' ')[0]: ",annotation.rstrip().split(' ')[0])
        #print("annotation.rstrip().split(' ')[1]: ",annotation.rstrip().split(' ')[1])
        #if len(annotation.rstrip().split(' ')==3):
            #image_file_0,image_file_1, labelname = annotation.rstrip().split(' ')
            #image_file = image_file_0+image_file_1
        #else:
            #image_file, labelname = annotation.rstrip().split(' ')
        image_file, labelname = annotation.rstrip().split(' ')   
        image_src = cv2.imread(image_file)
        image = letterbox(image_src,new_shape=(1280, 1280))[0]
        cv2.imwrite("./lettebox.jpg",image)
        
        # if 1:
        #     print("image_file: ", image_file)
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # note: Albumentation use rgb image format

        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]
        
        return image, int(self.label_map[labelname])
                
class ClassificationDataset_448_448(Dataset):
    
    def __init__(
        self,
        annotations_txt, 
        label_map_file,      
        augmentation=None,
        preprocessing=None 
    ):
        try:
            with open(annotations_txt, 'r') as f:
                self.annotations = f.readlines()
        except Exception as e:
            print("error: {}".format(e))  
        
        try:
            with open(label_map_file, "r") as f:
                self.label_map = json.load(f)
        except Exception as e:
            print("error: {}".format(e)) 

        # filter annotations,可能只需要训练一部分类别
        self.annotations = [annotation for annotation in self.annotations \
            if annotation.rstrip().split(' ')[-1] in self.label_map.keys()]

        self.augmentation = augmentation
        self.preprocessing = preprocessing      

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        #print("annotation.rstrip().split(' ')[0]: ",annotation.rstrip().split(' ')[0])
        #print("annotation.rstrip().split(' ')[1]: ",annotation.rstrip().split(' ')[1])
        #if len(annotation.rstrip().split(' ')==3):
            #image_file_0,image_file_1, labelname = annotation.rstrip().split(' ')
            #image_file = image_file_0+image_file_1
        #else:
            #image_file, labelname = annotation.rstrip().split(' ')
        image_file, labelname = annotation.rstrip().split(' ')   
        image_src = cv2.imread(image_file)
        image = letterbox(image_src,new_shape=(448, 448))[0]
        cv2.imwrite("./lettebox.jpg",image)
        
        # if 1:
        #     print("image_file: ", image_file)
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # note: Albumentation use rgb image format

        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]
        
        return image, int(self.label_map[labelname])
        
        
        
class ClassificationDataset_56_448(Dataset):
    
    def __init__(
        self,
        annotations_txt, 
        label_map_file,      
        augmentation=None,
        preprocessing=None 
    ):
        try:
            with open(annotations_txt, 'r') as f:
                self.annotations = f.readlines()
        except Exception as e:
            print("error: {}".format(e))  
        
        try:
            with open(label_map_file, "r") as f:
                self.label_map = json.load(f)
        except Exception as e:
            print("error: {}".format(e)) 

        # filter annotations,可能只需要训练一部分类别
        self.annotations = [annotation for annotation in self.annotations \
            if annotation.rstrip().split(' ')[-1] in self.label_map.keys()]

        self.augmentation = augmentation
        self.preprocessing = preprocessing      

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        #print("annotation.rstrip().split(' ')[0]: ",annotation.rstrip().split(' ')[0])
        #print("annotation.rstrip().split(' ')[1]: ",annotation.rstrip().split(' ')[1])
        #if len(annotation.rstrip().split(' ')==3):
            #image_file_0,image_file_1, labelname = annotation.rstrip().split(' ')
            #image_file = image_file_0+image_file_1
        #else:
            #image_file, labelname = annotation.rstrip().split(' ')
        image_file, labelname = annotation.rstrip().split(' ')   
        image_src = cv2.imread(image_file)
        hwc = image_src.shape
        if(hwc[2]==1):     
          image_src = cv2.cvtColor(image_src,cv2.CV_GRAY2BGR)  

        image = letterbox(image_src,new_shape=(448, 56))[0]
       

       # cv2.imwrite("./lettebox.jpg",image)
        
        # if 1:
        #     print("image_file: ", image_file)
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # note: Albumentation use rgb image format
        # print("channnel: ",image.shape)
        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        if self.preprocessing:
            image = self.preprocessing(image=image)["image"]
        
        return image, int(self.label_map[labelname])
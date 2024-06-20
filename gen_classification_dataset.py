# coding: utf-8

'''
    功能描述：由标注数据生成标准化的（可被ClassificationDataset类加载）分类数据集

    实现步骤：
        1. 逐目录切分数据集为train_目录名.txt, val_目录名.txt
        2. 合并为train.txt, val.txt
        3. 生成对应的json_map文件
    
    说明：
        1.label_data_root下各子目录数据的一致性和正确性都应该提前检查下
        2.label_data_root下各子目录建议用加样日期命名，各子目录下的子目录按类别名命名
'''

import os
import time
import random
import tqdm
import glob
import json
from utils.tools import *


class ClassificationDatasetPreparation(object):

    def __init__(
            self,
            label_data_root,  # 标注数据根目录
            label_map=None,
            trainset_ratio=0.9,
            shuffle=True,
            keyword=None,
    ):
        self.label_data_root = label_data_root
        self.label_map = label_map
        self.trainset_ratio = trainset_ratio
        self.shuffle = shuffle
        if keyword:
            self.keyword = keyword
        else:
            self.keyword = [None]

        self.dirs = os.listdir(self.label_data_root)
        self.num_classes = 0

    def gen_annotations(self, dataset_path):
        for idx in tqdm.tqdm(range(len(self.dirs))):
            dirname = self.dirs[idx]
            print(f"processing dir: {dirname}")
            annotations = self._get_subdir_annotations(os.path.join(self.label_data_root, dirname))
            train_size = round(self.trainset_ratio * len(annotations))
            train_txt = os.path.join(dataset_path, f"train_{dirname}.txt")
            val_txt = os.path.join(dataset_path, f"val_{dirname}.txt")
            self._save_annotations(annotations[0:train_size], train_txt)
            self._save_annotations(annotations[train_size:], val_txt)

        # 合并txt信息到train.txt, val.txt
        train_txt_files = glob.glob(os.path.join(dataset_path, "train_*.txt"))
        val_txt_files = glob.glob(os.path.join(dataset_path, "val_*.txt"))
        self._merge_txt_info(train_txt_files, os.path.join(dataset_path, "train.txt"))
        self._merge_txt_info(val_txt_files, os.path.join(dataset_path, "val.txt"))

        self._gen_label_map(dataset_path)

    def _get_subdir_annotations(self, path):
        annotations = []

        # 标注信息生成
        files = []
        for keyword in self.keyword:
            files.extend(get_file_list(path, keyword))
        files.sort()
        if self.shuffle:
            random.seed(1)
            random.shuffle(files)
        for file in files:
            dirname = os.path.basename(os.path.dirname(file))
            annotations.append(f"{file} {dirname}")

        return annotations

    def _save_annotations(self, annotations, annotations_file):
        with open(os.path.join(annotations_file), 'w') as fid:
            for idx in tqdm.tqdm(range(len(annotations))):
                fid.write(annotations[idx] + '\n')

    def _merge_txt_info(self, txt_files, save_txt):
        with open(save_txt, "w+") as fwriter:
            for txt_file in txt_files:
                with open(txt_file, "r") as freader:
                    info = freader.read()
                    fwriter.write(info)

    def _gen_label_map(self, dataset_path):
        if self.label_map is None:
            labels = set()
            for dir in self.dirs:
                dirs = os.listdir(os.path.join(self.label_data_root, dir))
                labels = labels | set(dirs)
            labels = list(labels)
            labels.sort()
            # print(labels)
            self.label_map = {label: str(idx) for idx, label in enumerate(labels)}
        print("generating label map: ", self.label_map)
        with open(os.path.join(dataset_path, 'label_map.json'), 'w') as f:
            json.dump(self.label_map, f)


if __name__ == "__main__":
    # 分类标注并整理好的数据集路径根目录
    # label_data_root = r'/mnt/10_AlgorithmData/ZNJT/ZhoneChe_sdg_blur_check/CleanData'
    # label_data_root = '/mnt/10_AlgorithmData/ZNJT/ZhongChe_sdg_integrity/CleanData'
    # label_data_root = '/data2/wangj/dataset/ZhongChe_sdg_integrity/CleanData'
    # label_data_root = r"/mnt/10_AlgorithmData/ZNJT/ShenYang_ZhuiYuanQu/LabelData"
    # keyword = [".jpg", ".png"]
    # label_map = {"fall": '0', "stand": '1'}
    # label_map = {"blur": "0", "clear": "1"} #不传label_map参数更通用，但是当类别存在增删改的情形时最好手动设置
    # label_map = {"normal": '0', "break": '1', "missing": '2', "deformation": '3'}

    # task:中车受电弓场景分类
    # label_data_root = "./tasks/CleanData"
    # keyword = [".jpg", ".png"]
    # label_map = {"blur": '0', "simple": '1', "complex": '2'}

    # task:T4的眼睛状态
    # label_data_root = "./eye_date_224_224/"
    # keyword = [".jpg", ".png"]
    # label_map = {"close": '0', "open": '1', "other": '2'}

    # task:探头图像的分类
    # label_data_root = "./train_datasets/classe_tt/"
    # keyword = [".jpg", ".png"]
    # label_map = {"good": '0', "watercols": '1', "blur": '2', "yiwu": '2'}

    # # task:中车受电弓完整性分类
    # label_data_root = '/mnt/10_AlgorithmData/ZNJT/ZhongChe_sdg_integrity/CleanData'
    # keyword = [".jpg", ".png"]
    # label_map = {"normal": '0', "break": '1', "missing": '2', "deformation": '3'}

    # # task:中车受电弓完模糊分类
    # label_data_root = '/mnt/10_AlgorithmData/ZNJT/ZhoneChe_sdg_blur_check/CleanData'
    # keyword = [".jpg", ".png"]
    # label_map = {"blur": "0", "clear": "1"}

    # 百国货币分类
    label_data_root = r"G:\piBu_STA_data\train_720x450_rs_512x320\jx_6class\class_raw/"
    keyword = [".jpg", ".png", "bmp"]
    label_map = {"cuokou": '0', "cuozong": '1', "duanjing": '2', "shuangjing": '3', "songjingjing": '4'}

    task_name = os.path.basename(os.path.dirname(label_data_root))

    dataset_path = os.path.join('./tasks', task_name, 'dataset')
    make_if_not_exit(dataset_path)

    dataset_preparation = ClassificationDatasetPreparation(label_data_root, keyword=keyword, label_map=label_map)
    # dataset_preparation = ClassificationDatasetPreparation(label_data_root, label_map=label_map)
    dataset_preparation.gen_annotations(dataset_path)

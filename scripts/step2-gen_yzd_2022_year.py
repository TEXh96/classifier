#coding: utf-8
import os
import cv2
from utils import *
from forward import *
import tqdm


rects = {
    "shanghai": [719, 613, 1031, 705],
    "nanjing": [459, 618, 685, 700],
}


def get_img_info(file):
    infos = file.split('_')
    loc = infos[0].split('-')[-1]
    
    return loc, infos[2]


if __name__ == '__main__':
    #root_path = '/mnt/10_AlgorithmData/ZNJT/wangj_temp/prob/20220311-shanghai'
    root_path='/mnt/10_data3_zhudao/上海/2022-1/'
    save_path = '/mnt/10_data3_zhudao/上海/2022-2/'#os.path.basename(root_path)
    os.makedirs(save_path, exist_ok=True)
    dirs = os.listdir(root_path)
    dirs = [os.path.join(root_path, dir) for dir in dirs if os.path.isdir(os.path.join(root_path, dir))]
    
    model_file = '/home/cbpm/duchao/wm/classifier/tasks/yzd/checkpoints/mobilenetv2_final.pth'
    labelmap_file = '/home/cbpm/wangj/project/zhudaokeji/prob/scripts/label_map.json'
    classifier = Classifier(model_file, labelmap_file)
    
    for idx in tqdm.tqdm(range(len(dirs))):
        dir = dirs[idx]
        print("dir: ",dir)
        
        loc = os.path.basename(dir)
        print("loc: ",loc)
        # import pdb
        # pdb.set_trace()
        img_files = get_file_list(dir)
        img_files.sort()
        print("img_files: ",img_files)
        # _, camID = get_img_info(os.path.basename(img_files[0]))
        # print("camID: ",camID)
        yzd = -1
        max_count = 20 #每个预置点最多存20帧
        for img_file in img_files:
            # print(img_file)
            print("camID: ",os.path.basename(img_file).split("_"))
            # import pdb
            # pdb.set_trace()
            
            camID = str(int(os.path.basename(img_file).split("_")[2]))
            print("camID: ",camID)
            # import pdb
            # pdb.set_trace()
            img = cv2.imread(img_file)
            rect = rects[loc]
            if img is None:
                continue
            img_roi = img[rect[1]:rect[3], rect[0]:rect[2], :].copy()
            label = classifier.forward(img_roi)
            if label > 0 and label!=yzd:
                yzd = label 
                counter = 0
            if yzd == -1:
                continue
            if counter >= max_count:
                continue
            path = os.path.join(save_path, loc, camID, str(yzd))
            os.makedirs(path, exist_ok=True)
            # os.chmod(path,777)
            name = f"{loc}_{camID}_{str(yzd)}_{os.path.basename(img_file)}"
            cv2.imwrite(os.path.join(path, name),img)
            # shutil.copy(img_file, os.path.join(path, name))
            counter += 1
    

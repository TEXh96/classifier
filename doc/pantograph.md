<!--
 * @Author: your name
 * @Date: 2021-03-17 14:54:06
 * @LastEditTime: 2021-03-18 15:17:08
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /classifier/doc/受电弓训练备忘.md
-->
# 备忘

**工程路径**

> * server03: /media/cbpm2016/D/wangj/pytorch/classifier

数据集路径：

细分类别：
    0：正常受电弓（normal_pantograph）
    1：弓角断裂(break)
    2：弓角缺失(missing)
    3：碳滑板变形(deformation)
    4: 异物入侵(foreign)

# 数据集制作流程
1. 原始视频转图像序列label
2. 数据类别整理标注labeled_data
3. 受电弓目标检测裁切受电弓区域（这里应该做适当外扩）raw_data
4. 按分类任务自定义的目录结构将数据合并到数据集中clean_data

# 训练及测试

```shell
source activate liaolong_torch1.4
python dataset/prepare_classification_dataset.py
python train_pantograph.py
```

# 模型转换及测试

xavier 
    fps32: 26.6ms 
    int8: 15.7ms 

待封装工程路径：
    mic-730ai@192.168.31.222 abc-123
    /home/mic-730ai/YJ_WORKDIR/sdg_classification


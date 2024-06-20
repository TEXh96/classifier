<!--
 * @Author: your name
 * @Date: 2022-03-15 18:48:19
 * @LastEditTime: 2022-03-15 18:49:49
 * @LastEditors: your name
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /prob/scripts/readme.md
-->
208: /home/cbpm/wangj/project/zhudaokeji/prob/scripts

source activate torch17_py36
cd /home/cbpm/wangj/project/zhudaokeji/prob/scripts
python step1-videos2images_multi_thread.py
python step2-gen_yzd.py


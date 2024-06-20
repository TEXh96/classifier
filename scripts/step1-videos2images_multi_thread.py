#coding: utf-8

import threading
from utils import *


def process_videos(video_files, image_save_path, interval, start_frame):
    # for idx in tqdm.tqdm(range(len(video_files))):
    for video_file in video_files:
        try:
            # video_file = video_files[idx]
            print("{} process: {}".format(threading.currentThread, video_file))
            video_to_images(video_file, image_save_path, interval=interval, start_frame=start_frame)
        except:
            continue


class myThread(threading.Thread):

    def __init__(self, thread_name, description, video_files, image_save_path=None, interval=50, start_frame=1):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.description = description
        self.video_files = video_files
        self.image_save_path = image_save_path
        self.interval = interval
        self.start_frame = 1

    def run(self):
        print("start thread: ", self.thread_name)
        process_videos(self.video_files, image_save_path=self.image_save_path, interval=self.interval, start_frame=self.start_frame)
        print("end thread: ", self.thread_name)


if __name__ == "__main__":
#    video_path = r'/mnt/10_AlgorithmData/ZNJT/wangj_temp/prob'
    video_path = r'/mnt/10_data3_zhudao/上海/2020/20201211-上海-数据整理/'
    save_path = r'/mnt/10_data3_zhudao/上海/2020-1/20201211-上海-数据整理/shanghai/'
    image_save_path = save_path + "_images"
    files = get_file_list(video_path)
    # print("files: ", files)
    video_files = files_filter(files, ['mkv'])
    print("video_files: ", video_files)
    video_files.sort()
    # video_files = video_files[49:]
    print(len(video_files))

    thread_count = 16 #16
    if len(video_files) < thread_count:
        thread_count = len(video_files)
    video_size = int(len(video_files)/thread_count + 0.5)
    print(video_size)
    videos_list = []
    jobs = []
    for i in range(thread_count):
        threadID = "Th"+str(i)
        start = i*video_size
        end = (i+1)*video_size
        if end > len(video_files):
            end = len(video_files)
        videos = video_files[start: end]
        th = myThread(threadID, "process videos", videos, image_save_path, interval=10)
        # th = myThread(threadID, "process videos", videos, interval=10)
        jobs.append(th)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

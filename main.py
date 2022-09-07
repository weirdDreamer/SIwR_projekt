import sys
import cv2
import numpy as np
import os
import re


def sorting_key(frame_no):
    return frame_no[1]


if __name__ == "__main__":
    # pobranie pierwszego parametru (ścieżki do katalogu z plikami)
    # z parametów uruchomieniowych
    base_path = sys.argv[1]
    # pobranie nazw poszczególnych klatek z filmu
    pic_names = os.listdir(base_path + '/frames')
    # sprawdzenie co wypluła funkcja
    # print(pic_names)

    pic_data = []
    for pic_name in pic_names:
        # print(pic_name)
        # print(pic_name[5:11])
        # print(int(pic_name[5:11]), '\n')
        pic_data.append((pic_name, int(pic_name[5:11])))

    # print(pic_data)
    pic_data_sorted = pic_data.copy()
    pic_data_sorted.sort(key=sorting_key)
    # print(pic_data_sorted)

    bbox_file_path = base_path + '/bboxes.txt'
    with open(bbox_file_path) as bbox_file:
        lines = bbox_file.readlines()
    bbox_file.close()

    video_bbox_data = []
    pic_bbox_data = {}
    bb_num = 0
    bb_counter = 0
    save_pic_data = False

    for line in lines:
        line_len = len(line)

        if 4 < line_len < 20:
            pic_bbox_data['name'] = line[:-1]

        if 4 > line_len:
            pic_bbox_data['count'] = bb_num = int(line[:-1])

        if 20 < line_len:
            if bb_counter < bb_num:
                pic_bbox_data['N' + str(bb_counter)] = [float(n) for n in line.split(' ')]
                bb_counter += 1

        if bb_counter == bb_num and bb_num > 0:
            video_bbox_data.append(pic_bbox_data)
            # print(pic_bbox_data)
            pic_bbox_data = {}
            bb_num = 0
            bb_counter = 0

    print('\n\n', video_bbox_data)
    # print('\n\n', pic_bbox_data)

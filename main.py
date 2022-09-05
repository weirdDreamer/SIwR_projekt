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
    print(pic_data_sorted)

    bbox_file_path = base_path + '/bboxes.txt'
    with open(bbox_file_path) as bbox_file:
        lines = bbox_file.readlines()
    bbox_file.close()

    video_pic_bbox_data = []
    pic_bbox_data = {}
    pic_bbox_num = 0
    bbox_read_flag = False
    read_no_of_bbox_flag = False

    for line in lines:
        # line_len = len(line)
        if read_no_of_bbox_flag:
            pic_bbox_data['No'] = int(line[:-1])
            read_no_of_bbox_flag = False
            print(pic_bbox_data)

            if pic_bbox_num < pic_bbox_data['No']:
                print(pic_bbox_data['No'])
                pic_bbox_num += 1
                pic_bbox_data['N'+str(pic_bbox_num)] = line

            if pic_bbox_num == pic_bbox_data['No']:
                video_pic_bbox_data.append(pic_bbox_data)
                pic_bbox_data = {}
                bbox_read_flag = False

        if not bbox_read_flag:
            x = re.search('jpg', line)
            if x is not None:
                pic_bbox_data['name'] = line[:-1]
                bbox_read_flag = True
                read_no_of_bbox_flag = True







        # if 10 < line_len < 20:
        #     if not pic_bbox_data:
        #         pic_bbox_data['name'] = line[:-1]
        #         video_pic_bbox_data.append(pic_bbox_data)
        #         pic_bbox_data = {}
        #     else:
        #         pic_bbox_data['name'] = line[:-1]
        # if 20 < line_len:
        #     # x_w x_h w h
        #     pass
        # if line_len < 4:
        #     pic_bbox_data['No'] = str(line)
        #     pass

    print('############', video_pic_bbox_data)
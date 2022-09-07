import sys
import cv2
import numpy as np
import os
import re


def sorting_key(frame_no):
    return frame_no[1]


class Frame:
    def __int__(self, base_path, name, bb_count, bb_pos_dim):

        self.img = cv2.imread(base_path+'/frames/'+name)
        self.img_name = name
        self.img_width = self.img.shape[0]
        self.img_height = self.img.shape[1]
        self.bbox_count = bb_count
        self.bbox_pos_dim_float = bb_pos_dim
        self.bbox_pos_dim_int = []
        for pos_dim in bb_pos_dim:
            pos_dim_int = [round(pos_dim)for pos_dim in pos_dim]
            self.bbox_pos_dim_int.append(pos_dim_int)
        self.bbox_cutted_out = []
        for pos_dim in self.bbox_pos_dim_int:
            x, y, w, h = pos_dim
            sub_img = self.img[x:x+w, y:y+h, :]
            self.bbox_cutted_out.append(sub_img)

    def get_bbox_num(self):
        return self.bbox_count

    def get_bbox_x_y_w_h_int(self, num):
        if num <= self.bbox_count:
            return self.bbox_pos_dim_int[num]
        else:
            return None

    def get_bbox_x_y_w_h_float(self, num):
        if num <= self.bbox_count:
            return self.bbox_pos_dim_float[num]
        else:
            return None


if __name__ == "__main__":
    # pobranie pierwszego parametru (ścieżki do katalogu z plikami)
    # z parametów uruchomieniowych
    base_path = sys.argv[1]
    # # pobranie nazw poszczególnych klatek z filmu
    # pic_names = os.listdir(base_path + '/frames')
    # # sprawdzenie co wypluła funkcja
    # # print(pic_names)
    #
    # pic_data = []
    # for pic_name in pic_names:
    #     # print(pic_name)
    #     # print(pic_name[5:11])
    #     # print(int(pic_name[5:11]), '\n')
    #     pic_data.append((pic_name, int(pic_name[5:11])))
    #
    # # print(pic_data)
    # pic_data_sorted = pic_data.copy()
    # pic_data_sorted.sort(key=sorting_key)
    # # print(pic_data_sorted)

    bboxes_file_path = base_path + '/bboxes.txt'
    with open(bboxes_file_path) as bboxes_file:
        lines = bboxes_file.readlines()
    bboxes_file.close()

    video_bbox_data = []
    pic_bbox_data = {}
    bb_num = 0
    bb_counter = 0
    n_equal_0_flag = False

    frame_proccesing_flag = False
    frames = []
    temp = -1
    for line in lines:
        line_len = len(line)

        if 4 < line_len < 20:
            pic_bbox_data['name'] = line[:-1]

        if 4 > line_len:
            pic_bbox_data['count'] = bb_num = int(line[:-1])
            if bb_num == 0:
                n_equal_0_flag = True
        if 20 < line_len:
            if bb_counter < bb_num:
                pic_bbox_data['N' + str(bb_counter)] = [float(n) for n in line.split(' ')]
                bb_counter += 1

        if (bb_counter == bb_num and bb_num > 0) or n_equal_0_flag:
            # video_bbox_data.append(pic_bbox_data)
            frame = 1 #Frame()
            frames.insert(0, frame)
            if len(frames) > 5:
                frames.pop(-1)
            pic_bbox_data = {}
            bb_num = 0
            bb_counter = 0
            n_equal_0_flag = False
            frame_proccesing_flag = True

        if frame_proccesing_flag:
            frame_proccesing_flag = False
            print(frames)


    print('\n\n', video_bbox_data)

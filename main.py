import sys
import cv2.cv2 as cv
import numpy as np
import os
import random
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation


class Frame:
    def __init__(self, base_path, name, bb_count, bb_pos_dim):
        self.img = cv.imread(base_path+'/frames/'+name)
        self.img_name = name
        self.img_width = self.img.shape[0]
        self.img_height = self.img.shape[1]
        self.bbox_count = bb_count
        self.bbox_pos_dim_float = bb_pos_dim
        self.bbox_pos_dim_int = None
        self.bboxes = None
        self.bboxes_marks = []

    def do_the_processig(self):
        self.bbox_pos_dim_int = []
        for bb_pos_dim in self.bbox_pos_dim_float:
            self.bbox_pos_dim_int.append([round(pos_dim) for pos_dim in bb_pos_dim])

        self.bboxes = []
        for bb_pos_dim in self.bbox_pos_dim_int:
            x, y, w, h = bb_pos_dim
            self.bboxes.append(self.img[y:y+h, x:x + w, :])

    def subtract_bg(self, background, subtractor):
        pass


def get_probability(curr_frame, prev_frame):
    graph = FactorGraph()
    if prev_frame.bbox_count == 0:

        mat_dim = prev_frame.bbox_count+1
        neighbour_nodes_mat = np.ones((mat_dim, mat_dim), dtype=float)
        np.fill_diagonal(neighbour_nodes_mat, 0.0)
        neighbour_nodes_mat[0, 0] = 1.0

    else:
        pass


if __name__ == "__main__":
    # pobranie pierwszego parametru (ścieżki do katalogu z plikami) z parametów uruchomieniowych
    base_path = sys.argv[1]
    # pobranie nazw poszczególnych klatek z filmu
    pic_names = os.listdir(base_path + '/frames')

    pic_names_len = len(pic_names)
    no_of_samples = int(pic_names_len*0.15)
    the_choosen_ones = random.sample(range(0, pic_names_len-1), no_of_samples)

    frames_bg = []
    for i in the_choosen_ones:
        frames_bg.append(cv.imread(base_path+'/frames/'+pic_names[i]))

    viedo_backbround = np.median(frames_bg, axis=0).astype(dtype=np.uint8)
    cv.imshow('bg', viedo_backbround)
    # cv.wait

    bboxes_file_path = base_path + '/bboxes.txt'
    with open(bboxes_file_path) as bboxes_file:
        lines = bboxes_file.readlines()
    bboxes_file.close()

    video_bbox_data = []
    pic_bbox_data = {'pos_dim': []}
    bb_num = 0
    bb_counter = 0
    n_equal_0_flag = False

    frame_proccesing_flag = False
    frames_history = []

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
                # pic_bbox_data['N' + str(bb_counter)] = [float(n) for n in line.split(' ')]
                pic_bbox_data['pos_dim'].append([float(n) for n in line.split(' ')])
                bb_counter += 1

        if (bb_counter == bb_num and bb_num > 0) or n_equal_0_flag:
            # video_bbox_data.append(pic_bbox_data)
            # print(pic_bbox_data)
            frame = Frame(base_path, pic_bbox_data['name'], pic_bbox_data['count'], pic_bbox_data['pos_dim'])
            frames_history.append(frame)

            # trim frames list to given lenght
            if len(frames_history) > 3:
                frames_history.pop(1)

            pic_bbox_data = {'pos_dim': []}
            bb_num = 0
            bb_counter = 0
            n_equal_0_flag = False
            frame_proccesing_flag = True

        if frame_proccesing_flag:
            frame_proccesing_flag = False

    for frame in frames_history:
        cv.imshow('vid', frame.img)
        cv.waitKey()
        pass
    # print('\n\n', video_bbox_data)
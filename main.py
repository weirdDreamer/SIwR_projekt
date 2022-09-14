import sys
import cv2.cv2 as cv
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from itertools import combinations


class Frame:
    def __init__(self, base_path, name, bb_count, bb_pos_dim):
        self.img = cv.imread(base_path + '/frames/' + name)
        self.img_name = name
        self.img_width = self.img.shape[0]
        self.img_height = self.img.shape[1]
        self.bbox_count = bb_count
        self.bbox_pos_dim_float = bb_pos_dim
        # other variable initiation
        self.frame_processed_flag = False
        self.bbox_pos_dim_int = []
        self.bboxes = []
        self.bboxes_trimmed = []
        self.bboxes_hist = []
        self.bboxes_hist_avg = []
        self.bboxes_trimmed_hist = []
        self.bboxes_trimmed_hist_avg = []

    def process_the_frame(self):  # , background):
        if not self.frame_processed_flag:
            for bb_pos_dim in self.bbox_pos_dim_float:
                self.bbox_pos_dim_int.append([round(pos_dim) for pos_dim in bb_pos_dim])

            trim_f = 0.15 / 2

            for bb_pos_dim in self.bbox_pos_dim_int:
                # get position and dimentions out of list
                x, y, w, h = bb_pos_dim
                # put cuted out bounding boxes to list
                bbox = self.img[y:y + h, x:x + w, :]
                self.bboxes.append(bbox)

                d_h = int(h * trim_f)
                d_w = int(w * trim_f)
                bbox_trim = self.img[y + d_h:y + h - d_h, x + d_w:x + w - d_w, :]
                self.bboxes_trimmed.append(bbox_trim)

                bbox_hist = cv.calcHist([bbox], [0], None, [256], [0, 256])
                self.bboxes_hist.append(bbox_hist)

                bbox_trimmed_hist = cv.calcHist([bbox_trim], [0], None, [256], [0, 256])
                self.bboxes_trimmed_hist.append(bbox_trimmed_hist)
            self.frame_processed_flag = True


def get_probability(curr_frame, prev_frame):
    hist_diff_avg_list = []
    for curr_hist in curr_frame.bboxes_trimmed_hist:
        hist_diff_avg_sublist = []
        for prev_hist in prev_frame.bboxes_trimmed_hist:
            hist_diff = abs(curr_hist - prev_hist)
            hist_diff_avg = np.average(hist_diff[:-2])
            divider = abs(np.average(prev_hist) + np.average(curr_hist))
            hist_diff_avg_sublist.append(hist_diff_avg / divider)
        hist_diff_avg_list.append(hist_diff_avg_sublist)

    output_str = ''
    output_list = []
    if prev_frame.bbox_count > 0 and curr_frame.bbox_count > 0:
        data_in_vec_list = []
        nodes_nms = []
        for bb_idx in range(curr_frame.bbox_count):
            data_in_vec_sublist = []
            for idx, hist_avg in enumerate(hist_diff_avg_list[bb_idx]):
                data = 1 - hist_avg
                data_in_vec_sublist.append(data)
            data_in_vec_list.append(data_in_vec_sublist)
            nodes_nms.append('bb' + str(bb_idx))

        graph = FactorGraph()
        graph.add_nodes_from(nodes_nms)

        for idx in range(curr_frame.bbox_count):
            prob_mtx = [0.6] + data_in_vec_list[idx]
            df = DiscreteFactor([nodes_nms[idx]], [prev_frame.bbox_count + 1], prob_mtx)
            graph.add_factors(df)

            graph.add_edge(nodes_nms[idx], df)

        nms_comb = combinations(nodes_nms, 2)

        link_mtrx_dim = prev_frame.bbox_count + 1
        node_link_mtrx = np.ones((link_mtrx_dim, link_mtrx_dim), dtype=float)
        np.fill_diagonal(node_link_mtrx, 0.0)
        node_link_mtrx[0, 0] = 1.0

        for idx, pair in enumerate(nms_comb):
            # print(pair,'\n')
            df_link = DiscreteFactor(pair, [link_mtrx_dim, link_mtrx_dim], node_link_mtrx)
            graph.add_factors(df_link)
            graph.add_edge(pair[0], df_link)
            graph.add_edge(pair[1], df_link)

        propagation = BeliefPropagation(graph)
        output = propagation.map_query(variables=nodes_nms, show_progress=False)
        for out in output:
            output_str += str(output[out] - 1) + ' '
            output_list.append(output[out] - 1)

    elif prev_frame.bbox_count == 0 and curr_frame.bbox_count > 0:
        for idx in curr_frame.bbox_count:
            output_str += '-1 '
            output_list.append(-1)
    return output_str, output_list


if __name__ == "__main__":
    # get path to directory from program launch arguments
    base_path = sys.argv[1]
    # add file name to directory path
    bboxes_file_path = base_path + '/bboxes.txt'

    # read file into list of lines(strings)
    with open(bboxes_file_path) as bboxes_file:
        lines = bboxes_file.readlines()
    bboxes_file.close()

    # initiation of variables used in main loop
    pic_bbox_data = {'pos_dim': []}  # set to store data chunk
    bb_num = 0
    bb_counter = 0
    n_equal_0_flag = False
    start_up_flag = True
    frames_memory = []
    output_list = []  # list to store output as int values in testing purpose

    # main loop
    for line in lines:
        # get lenght of current line
        line_len = len(line)

        #
        if 4 < line_len < 20:
            pic_bbox_data['name'] = line[:-1]

        if 4 > line_len:
            pic_bbox_data['count'] = bb_num = int(line[:-1])
            if bb_num == 0:
                n_equal_0_flag = True

        if 20 < line_len:
            if bb_counter < bb_num:
                pic_bbox_data['pos_dim'].append([float(n) for n in line.split(' ')])
                bb_counter += 1

        if (bb_counter == bb_num and bb_num > 0) or n_equal_0_flag:

            frame = Frame(base_path, pic_bbox_data['name'], pic_bbox_data['count'], pic_bbox_data['pos_dim'])
            frames_memory.append(frame)

            # trim frames list to given lenght
            if len(frames_memory) > 3:
                frames_memory.pop(-4)
            frame.process_the_frame()
            output_str = ''
            output_sublist = []
            if not start_up_flag:
                output_str, output_sublist = get_probability(frame, frames_memory[-2])
            else:
                for idx in range(frame.bbox_count):
                    output_str += '-1 '
                    output_sublist.append(-1)
                start_up_flag = False

            print(output_str[:-1])
            output_list.append(output_sublist)
            # overwrite data
            pic_bbox_data = {'pos_dim': []}
            bb_num = 0
            bb_counter = 0
            n_equal_0_flag = False

    testing_flag = True
    if testing_flag:
        bboxes_gt_file_path = base_path + '/bboxes_gt.txt'
        with open(bboxes_gt_file_path) as gt_bboxes_file:
            gt_lines = gt_bboxes_file.readlines()
        gt_bboxes_file.close()

        gt_output = []
        sublist = []
        counter = 0
        no_of_bb = -1
        for line in gt_lines:
            line_len = len(line)

            if 4 > line_len:
                no_of_bb = int(str(line[:-1]))

            if 20 < line_len:
                if counter < no_of_bb:
                    sublist.append(int(line.split(' ')[0]))
                counter += 1

            if counter == no_of_bb:
                gt_output.append(sublist)
                no_of_bb = -1
                counter = 0
                sublist = []

        counter = 0
        divierd = 0
        for out_idx, out in enumerate(output_list):
            for val_idx, val in enumerate(out):
                divierd += 1
                if val == gt_output[out_idx][val_idx]:
                    counter += 1

        print('calc output len: ', len(output_list), ' gt output len: ', len(gt_output))
        print(counter, divierd)
        print('accuracy: ', float(counter / divierd))

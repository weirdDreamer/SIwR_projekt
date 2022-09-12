import sys
import cv2.cv2 as cv
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from  itertools import combinations

# TODO Jakość kodu i raport (1/5)
# TODO Brak raportu.
# TODO Projekt niedokończony.

# TODO Skuteczność śledzenia 0.0 (0/5)
# TODO [0.00, 0.0] - 0.0
# TODO (0.0, 0.1) - 0.5
# TODO [0.1, 0.2) - 1.0
# TODO [0.2, 0.3) - 1.5
# TODO [0.3, 0.4) - 2.0
# TODO [0.4, 0.5) - 2.5
# TODO [0.5, 0.6) - 3.0
# TODO [0.6, 0.7) - 3.5
# TODO [0.7, 0.8) - 4.0
# TODO [0.8, 0.9) - 4.5
# TODO [0.9, 1.0) - 5.0

# Stderr not empty
# niepoprawne wyjście: oczekiwana liczba linii = 400, otrzymana liczba linii = 3

class Frame:
    def __init__(self, base_path, name, bb_count, bb_pos_dim):
        self.img = cv.imread(base_path+'/frames/'+name)
        self.img_name = name
        self.img_width = self.img.shape[0]
        self.img_height = self.img.shape[1]
        self.bbox_count = bb_count
        self.bbox_pos_dim_float = bb_pos_dim
        # other variable initiation
        self.procesed = False
        self.bbox_pos_dim_int = []
        self.bboxes = []
        self.bboxes_trimmed = []
        self.bboxes_hist = []
        self.bboxes_hist_avg = []
        self.bboxes_trimmed_hist = []
        self.bboxes_trimmed_hist_avg = []

    def process_the_frame(self):  # , background):

        for bb_pos_dim in self.bbox_pos_dim_float:
            self.bbox_pos_dim_int.append([round(pos_dim) for pos_dim in bb_pos_dim])

        trim_f = 0.15 / 2

        for bb_pos_dim in self.bbox_pos_dim_int:
            # get position and dimentions out of list
            x, y, w, h = bb_pos_dim
            # put cuted out bounding boxes to list
            bbox = self.img[y:y+h, x:x + w, :]
            self.bboxes.append(bbox)

            d_h = int(h * trim_f)
            d_w = int(w * trim_f)
            bbox_trim = self.img[y + d_h:y + h - d_h, x + d_w:x + w - d_w, :]
            self.bboxes_trimmed.append(bbox_trim)

            bbox_hist = cv.calcHist([bbox], [0], None, [256], [0, 256])
            self.bboxes_hist.append(bbox_hist)
            self.bboxes_hist_avg.append(np.average(bbox_hist))

            bbox_trimmed_hist = cv.calcHist([bbox_trim], [0], None, [256], [0, 256])
            self.bboxes_trimmed_hist.append(bbox_trimmed_hist)
            self.bboxes_trimmed_hist_avg.append(np.average(bbox_trimmed_hist))
        self.procesed = True

def get_probability(curr_frame, prev_frame):
    hist_diff_avg_list = []
    for curr_hist in curr_frame.bboxes_trimmed_hist:
        hist_diff_avg_sublist = []
        for prev_hist in prev_frame.bboxes_trimmed_hist:
            hist_diff = abs(curr_hist-prev_hist)
            hist_diff_avg = np.average(hist_diff[:-2])
            divider = abs(np.average(prev_hist)+np.average(curr_hist))
            # print(hist_diff_avg/divider)

            hist_diff_avg_sublist.append(hist_diff_avg/divider)
        hist_diff_avg_list.append(hist_diff_avg_sublist)
    # TODO Program nie powinien nic wypisywac, oprocz wynikow.
    print('curr ', curr_frame.bbox_count, ' prev ', prev_frame.bbox_count)
    print(hist_diff_avg_list)
    # TODO W przypadku braku bboxów program powinien wypisać pustą linię.
    if prev_frame.bbox_count > 0 and curr_frame.bbox_count > 0:

        data_in_vec_list = []
        nodes_nms = []

        for bb_idx in range(curr_frame.bbox_count):
            data_in_vec_sublist = []
            for idx, hist_avg in enumerate(hist_diff_avg_list[bb_idx]):
                data = 1-hist_avg
                data_in_vec_sublist.append(data)
            data_in_vec_list.append(data_in_vec_sublist)
            nodes_nms.append('bb' + str(bb_idx))
        print(data_in_vec_list)

        graph = FactorGraph()
        graph.add_nodes_from(nodes_nms)

        for idx in range(curr_frame.bbox_count):
            prob_mtx = data_in_vec_list[idx].insert(0, 0.6)

            df = DiscreteFactor([nodes_nms[idx]], [prev_frame.bbox_count+1], prob_mtx)
            graph.add_factors(df)

            graph.add_edge((nodes_nms[idx], df))

        nms_comb = combinations(nodes_nms, 2)

        link_mtrx_dim = prev_frame.bbox_count + 1
        node_link_mtrx = np.ones((link_mtrx_dim, link_mtrx_dim), dtype=float)
        np.fill_diagonal(node_link_mtrx, 0.0)
        node_link_mtrx[0, 0] = 1.0

        for idx, pair in enumerate(nms_comb):
            print(pair)
            df_link = DiscreteFactor(pair, [link_mtrx_dim, link_mtrx_dim], node_link_mtrx)
            graph.add_factors(df_link)
            graph.add_edge((nms_comb[0], df_link))
            graph.add_edge((nms_comb[1], df_link))

        graph.check_model()

        propagation = BeliefPropagation(graph)

        # TODO Brakuje obliczenia i wypisania wyniku.
        # propagation.map_query()

    else:
        pass


if __name__ == "__main__":
    # pobranie pierwszego parametru (ścieżki do katalogu z plikami) z parametów uruchomieniowych
    base_path = sys.argv[1]
    # pobranie nazw poszczególnych klatek z filmu
    pic_names = os.listdir(base_path + '/frames')

    # pic_names_len = len(pic_names)
    # no_of_samples = int(pic_names_len*0.15)
    # the_choosen_ones = random.sample(range(0, pic_names_len-1), no_of_samples)
    #
    # frames_bg = []
    # for i in the_choosen_ones:
    #     frames_bg.append(cv.imread(base_path+'/frames/'+pic_names[i]))
    #
    # viedo_backbround = np.median(frames_bg, axis=0).astype(dtype=np.uint8)
    # cv.imshow('bg', viedo_backbround)
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

    # TODO Dlaczego tylko 50 pierwszych linii?
    lines = lines[:50]# pfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff

    for line in lines:
        line_len = len(line)

        # TODO Lepiej wykorzystać kolejność danych w pliku, a nie numery linii (ryzykowne, jeśli coś się zmieni np. w nazwach).
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

            # TODO Dlaczego tylko 3 klatki?
            # trim frames list to given lenght
            if len(frames_history) > 3:
                frames_history.pop(-4)

            pic_bbox_data = {'pos_dim': []}
            bb_num = 0
            bb_counter = 0
            n_equal_0_flag = False
            frame_proccesing_flag = True

        if frame_proccesing_flag:
            frame_proccesing_flag = False

    for idx, frame in enumerate(frames_history):
        # frame.process_the_frame()
        # cv.imshow('frame', frame.img)
        #
        # for i in range(frame.bbox_count):
        #     cv.imshow('bb', frame.bboxes[i])
        #     cv.imshow('bb_tr', frame.bboxes_trimmed[i])
        #
        #     # plt.subplot(211)
        #     # plt.plot(frame.bboxes_hist[i])
        #     # plt.subplot(212)
        #     # plt.plot(frame.bboxes_trimmed_hist[i])
        #     # plt.show()
        #     print('boxes hist:', frame.bboxes_hist_avg[i])
        #     print('boxes trimmed hist:', frame.bboxes_trimmed_hist_avg[i])
        #     cv.waitKey()
        if idx > 0:
            frames_history[idx - 1].process_the_frame()
            frame.process_the_frame()
            get_probability(frame, frames_history[idx-1])


        pass
    # print('\n\n', video_bbox_data)
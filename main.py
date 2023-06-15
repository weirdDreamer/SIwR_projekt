import sys
import cv2 as cv
import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from itertools import combinations


class Frame:
    # class used for data storing and processing data

    def __init__(self, base_path, name, bb_count, bb_pos_dim):
        # load basic variables
        self.img = cv.imread(base_path + '/frames/' + name)
        self.img_name = name
        self.bbox_count = bb_count
        self.bbox_pos_dim_float = bb_pos_dim
        # other variable initialization
        self.frame_processed_flag = False
        self.bbox_pos_dim_int = []
        self.bboxes = []
        self.bboxes_trimmed = []
        self.bboxes_hist = []
        self.bboxes_hist_avg = []
        self.bboxes_trimmed_hist = []
        self.bboxes_trimmed_hist_avg = []

    def process_the_frame(self):
        # function used for processing existing data and extracting additional data from an image

        # avoidance of double data processing
        if not self.frame_processed_flag:
            # convert float values of bb's position and dimensions to integer
            for bb_pos_dim in self.bbox_pos_dim_float:
                self.bbox_pos_dim_int.append([round(pos_dim) for pos_dim in bb_pos_dim])

            # prepare trimming factor (15%)
            trim_f = 0.15 / 2

            for bb_pos_dim in self.bbox_pos_dim_int:
                # get position and dimensions out of list
                x, y, w, h = bb_pos_dim
                # put cutted out bounding boxes to list
                bbox = self.img[y:y + h, x:x + w, :]
                self.bboxes.append(bbox)

                # calculate offset
                d_h = int(h * trim_f)
                d_w = int(w * trim_f)
                # get trimmed bounding box
                bbox_trim = self.img[y + d_h:y + h - d_h, x + d_w:x + w - d_w, :]
                self.bboxes_trimmed.append(bbox_trim)

                # calculate histogram of bounding box
                bbox_hist = cv.calcHist([bbox], [0], None, [256], [0, 256])
                self.bboxes_hist.append(bbox_hist)

                # calculate histogram of trimmed bounding box
                bbox_trimmed_hist = cv.calcHist([bbox_trim], [0], None, [256], [0, 256])
                self.bboxes_trimmed_hist.append(bbox_trimmed_hist)
            # change flag
            self.frame_processed_flag = True


def get_probability(curr_frame, prev_frame):
    # initialization of variable
    hist_diff_avg_list = []

    # calculate factors to feed factor graph
    for curr_hist in curr_frame.bboxes_trimmed_hist:
        hist_diff_avg_sublist = []
        for prev_hist in prev_frame.bboxes_trimmed_hist:
            # histogram subtraction
            hist_diff = abs(curr_hist - prev_hist)
            # average of hist. difference (expect extremely saturated colors)
            hist_diff_avg = np.average(hist_diff[:-2])
            # calculate divider for normalization
            divider = abs(np.average(prev_hist) + np.average(curr_hist))
            # add normalized value to list
            hist_diff_avg_sublist.append(hist_diff_avg / divider)
        hist_diff_avg_list.append(hist_diff_avg_sublist)

    # initialization of return values
    output_str = ''
    output_list = []
    # make sure that bounding boxes exist
    if prev_frame.bbox_count > 0 and curr_frame.bbox_count > 0:
        # initialization
        data_in_vec_list = []
        nodes_nms = []
        # crate and fill input vector
        for bb_idx in range(curr_frame.bbox_count):
            data_in_vec_sublist = []
            for hist_avg in hist_diff_avg_list[bb_idx]:
                # negate difference to similarity
                data = 1.0 - hist_avg
                data_in_vec_sublist.append(data)
            data_in_vec_list.append(data_in_vec_sublist)
            # crate nodes names and add them to list
            nodes_nms.append('bb' + str(bb_idx))

        # crete graph
        graph = FactorGraph()
        graph.add_nodes_from(nodes_nms)

        # create grph's edges and factors for nodes
        for idx in range(curr_frame.bbox_count):
            # add possibility that bounding box is new
            prob_vec = [0.49] + data_in_vec_list[idx]  # 47 89.05 48 89.315 49 89.35 495 89.279 50 89.32 60 87.67
            # create node factor with one greater cardinality using vector of probability
            df = DiscreteFactor([nodes_nms[idx]], [prev_frame.bbox_count + 1], prob_vec)
            # add nodes factors to graph
            graph.add_factors(df)
            # add edges to graph
            graph.add_edge(nodes_nms[idx], df)

        # get all combinations of nodes (sorted and without repeating)
        nms_comb = combinations(nodes_nms, 2)
        # prepare matrix being link between nodes
        # matrix size (one greater for probability that bb is mew)
        link_mtrx_size = prev_frame.bbox_count + 1
        # create matrix linking nodes (filled with 1 expect non-first elements of main diagonal)
        node_link_mtrx = np.ones((link_mtrx_size, link_mtrx_size), dtype=float)
        np.fill_diagonal(node_link_mtrx, 0.0)
        node_link_mtrx[0, 0] = 1.0

        # craete factors linking nodes
        for idx, pair in enumerate(nms_comb):
            df_link = DiscreteFactor(pair, [link_mtrx_size, link_mtrx_size], node_link_mtrx)
            # adding factors to graph
            graph.add_factors(df_link)
            graph.add_edge(pair[0], df_link)
            graph.add_edge(pair[1], df_link)

        # performing inference using Belief Propagation method
        propagation = BeliefPropagation(graph)
        # get output by node names without printing progress
        output = propagation.map_query(variables=nodes_nms, show_progress=False)

        # preparation output string
        for out in output:
            # adapting the output to the required format
            output_str += str(output[out] - 1) + ' '
            output_list.append(output[out] - 1)

    # set -1 for every bb when previous frame does not contain bb's
    elif prev_frame.bbox_count == 0 and curr_frame.bbox_count > 0:
        for num in range(curr_frame.bbox_count):
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

    # initialization of 'circulating' variables used in main loop
    pic_bbox_data = {'pos_dim': []}  # set to store data chunk
    bb_num = 0
    bb_counter = 0
    n_equal_0_flag = False
    start_up_flag = True
    frames_memory = []
    output_list = []  # list to store output as integer in testing purpose

    # main loop
    for line in lines:
        # get length of current line
        line_len = len(line)

        # interpret certain line length as line with frame's name, it's sensitive for name length
        if 4 < line_len < 20:
            # add frame's name to set
            pic_bbox_data['name'] = line[:-1]

        # interpret certain line length as line with number of bounding boxes (less than 999)
        if 4 > line_len:
            # add bounding boxes count to frame's set
            pic_bbox_data['count'] = bb_num = int(line[:-1])
            # raise's flag when there is no bb in picture
            if bb_num == 0:
                n_equal_0_flag = True

        # interpret certain line length as line with position and dimensions of bounding box
        if 20 < line_len:
            # check if number of loaded bb's position lines is lower than bb's number
            if bb_counter < bb_num:
                # add position and dimensions to frame's set
                pic_bbox_data['pos_dim'].append([float(n) for n in line.split(' ')])
                # increment counter
                bb_counter += 1

        # check if current frame's is fully loaded
        if (bb_counter == bb_num and bb_num > 0) or n_equal_0_flag:
            # load data into the Frame class
            frame = Frame(base_path, pic_bbox_data['name'], pic_bbox_data['count'], pic_bbox_data['pos_dim'])
            # add data to 'memory' list
            frames_memory.append(frame)

            # trim frames list to given length
            if len(frames_memory) > 3:
                frames_memory.pop(-4)

            # do processing on current frame
            frame.process_the_frame()
            # initialization of output string
            output_str = ''
            output_sublist = []  # sublist to store output as integer

            # skip processing of first frame
            if not start_up_flag:
                # get probability based on current and previous frame
                output_str, output_sublist = get_probability(frame, frames_memory[-2])
            else:
                # fill output with -1 (value for new bb's) for firs
                for idx in range(frame.bbox_count):
                    output_str += '-1 '
                    output_sublist.append(-1)
                    # "reset" flag
                start_up_flag = False
            # print output string (expect last whitespace)
            print(output_str[:-1])
            # add data to test string
            output_list.append(output_sublist)

            # reset 'circulating' valuable
            pic_bbox_data = {'pos_dim': []}
            bb_num = 0
            bb_counter = 0
            n_equal_0_flag = False

    # accuracy test
    testing_mode_flag = False #True for testing
    if testing_mode_flag:
        # add file name to directory path
        bboxes_gt_file_path = base_path + '/bboxes_gt.txt'

        # read verification data
        with open(bboxes_gt_file_path) as bboxes_gt_file:
            gt_lines = bboxes_gt_file.readlines()
        bboxes_gt_file.close()

        # initialization of 'circulating' variables
        gt_output = []
        sublist = []
        counter = 0
        no_of_bb = -1

        # load verification data
        for line in gt_lines:
            # get line length
            line_len = len(line)
            # get number of bb
            if 4 > line_len:
                no_of_bb = int(str(line[:-1]))

            # get bb marking
            if 20 < line_len:
                if counter < no_of_bb:
                    # add marking to sublist
                    sublist.append(int(line.split(' ')[0]))
                # increment counter
                counter += 1

            if counter == no_of_bb:
                # add collected data to list
                gt_output.append(sublist)
                # reset 'circulating' variables
                no_of_bb = -1
                counter = 0
                sublist = []

        # variables for accuracy calculation
        counter = 0
        divider = 0
        # compare estimated data to verification data
        for out_idx, out in enumerate(output_list):
            for val_idx, val in enumerate(out):
                print('est:', out, 'gt:', gt_output[out_idx])
                divider += 1
                # compare values
                if val == gt_output[out_idx][val_idx]:
                    counter += 1
        print('accuracy: ', float(counter / divider))

from argparse import ArgumentParser
import tensorflow as tf
import time
# from math import pow
import os
from random import shuffle
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
from generator import generator_test_region, generator_img_region
from archs import R_Net


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    # print(not_initialized_vars)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

parser = ArgumentParser()
parser.add_argument("-s", "--save-log", help="save to train_log", dest="save_log", default="1")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.1")
args = parser.parse_args()

save_log = os.path.join('train_log',
                        args.save_log)
save_folder = os.path.join(save_log, 'models')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    net = R_Net.R_Net()

    input_img = tf.placeholder(tf.float32,
                               [None, None, None, 3])
    feature_map = net.pr(input_img)

    model_pr = tf.train.Saver()
    model_pr.restore(sess, os.path.join(save_folder,
                                        str(10000) + 'save_net.ckpt'))

    ge = generator_img_region.generator(json_path='/data/train_label.json',
                                        image_path='/data/img_pyramids/',
                                        image_size=24,
                                        batch=int(10),
                                        image_from_each_face=3,
                                        dy=[5, 11, 20])

    initialize_uninitialized(sess)

    while True:

        begin_time = time.time()
        imgs, \
        labs, \
        com_labs, \
        ori_imgs, \
        ori_labels, \
        crops \
            = ge.__next__()

        if len(imgs) == 0:
            break
        tic = time.time()
        output = sess.run(feature_map,
                          feed_dict={input_img: imgs})
        for i in range(output.shape[0]):
            output_p = output[i, :]
            if output_p[0] > output_p[1]:
                labeln = output_p[2:]
                # label = np.zeros([4], np.float)
                # print(labeln)
                # label[0] = labeln[2] * ori_img[i].size[0]#/48*48
                # label[1] = labeln[3] * ori_img[i].size[0]#/48*48
                # label[2] = math.exp(labeln[2]) * ori_img[i].size[0]#/48*48
                # label[3] = math.exp(labeln[3]) * ori_img[i].size[0]#/48*48

                print(labeln)
                labeln = labeln * (crops[i][2] - crops[i][0]) / 24

                n_crops = [crops[i][0] + labeln[0], crops[i][1] + labeln[1],
                           crops[i][0] + labeln[2], crops[i][1] + labeln[3]]

                draw = ImageDraw.Draw(ori_imgs[i])
                draw.rectangle(((crops[i][0], crops[i][1]), (crops[i][2], crops[i][3])), outline='black')
                draw.rectangle(((n_crops[0], n_crops[1]), (n_crops[2], n_crops[3])), outline='red')
                ori_imgs[i].show()
                input()

        end_time = time.time()
        print('spend time: %.3f,' % (end_time - begin_time))

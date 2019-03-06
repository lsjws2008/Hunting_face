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
from generator import generator_test_region
from archs import P_Net


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    # print(not_initialized_vars)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


parser = ArgumentParser()
parser.add_argument("-s", "--save-log", help="save to train_log", dest="save_log", default="0")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.1")
args = parser.parse_args()

save_log = os.path.join('train_log',
                        args.save_log)
save_folder = os.path.join(save_log, 'models')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    net = P_Net.P_Net()

    input_img = tf.placeholder(tf.float32,
                               [None, None, None, 3])
    feature_map = net.P_pr(input_img)

    model_pr = tf.train.Saver()
    model_pr.restore(sess, os.path.join(save_folder,
                                        str(9000) + 'save_net.ckpt'))

    ge = generator_test_region.generator_test(image_path='/data/images/',
                        image_size=12,
                        batch=1,
                        image_from_each_face=2)

    initialize_uninitialized(sess)

    while True:
        begin_time = time.time()

        img, label, ori_img = ge.__next__()
        if len(img) == 0:
            break
        tic = time.time()
        output = sess.run(feature_map,
                          feed_dict={input_img: img})
        for i in range(output.shape[0]):
            output_p = output[i, :, :, :]
            output_p = np.squeeze(output_p)
            if output_p[0] > output_p[1]:
                print(output_p)
                draw = ImageDraw.Draw(ori_img[i])
                labeln = output_p[2:]
                labeln *= ori_img[i].size[0]/12
                draw.rectangle(((labeln[0], labeln[1]), (labeln[2], labeln[3])))
                ori_img[i].show()
                input()
        end_time = time.time()
        print('spend time: %.3f,' % (end_time - begin_time))

from argparse import ArgumentParser
import tensorflow as tf
from archs import P_Net, losses, O_Net, R_Net
import time
# from math import pow
import os
from random import shuffle
import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
from generator import generator_test_region, generator_R_O_region


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


batch = 3

parser = ArgumentParser()
parser.add_argument("-S", "--save-log", help="save to train_log", dest="save_log", default="3")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.4")

args = parser.parse_args()
save_log = os.path.join('train_log',
                        args.save_log)

save_folder = os.path.join(save_log, 'models')

if not os.path.isdir(save_log):
    os.mkdir(save_log)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

output_num = 10

with tf.Session(config=config) as sess:

    r_net = R_Net.R_Net()

    o_net = O_Net.O_Net()

    ge = generator_test_region.generator_test(image_path='/data/images/',
                                              image_size=24,
                                              batch=1,
                                              image_from_each_face=3,
                                              dy=[5, 11, 20])

    r_o_ge = generator_R_O_region.generator_r_o()

    img = tf.placeholder(tf.float32,
                         [None, None, None, 3])
    lab = tf.placeholder(tf.float32,
                         [None, output_num])
    com_lab = tf.placeholder(tf.float32,
                             [None, output_num])
    lr = tf.placeholder(tf.float32,
                        [None])

    r_output = r_net.pr(img)

    o_output = o_net.pr(img)
    output = tf.reshape(o_output, [-1, 10])

    model_pr = tf.train.Saver()
    model_pr.restore(sess, os.path.join(save_folder,
                                        str(4000) + 'save_net.ckpt'))

    initialize_uninitialized(sess)

    print('begin: ')
    while True:
        begin_time = time.time()

        n_imgs = []
        n_ori_imgs = []
        n_ori_crops = []
        n_r_crops = []

        imgs, \
        labs, \
        ori_imgs, \
        crops \
            = ge.__next__()
        # labs = [i[1] for i in labs]
        r_out = sess.run(r_output,
                         feed_dict={img: imgs})

        for i in range(r_out.shape[0]):
            if r_out[i][0] > r_out[i][1]:
                n_out = r_out[i] * (crops[i][2] - crops[i][0]) / 24

                n_crops = [crops[i][0] + n_out[2], crops[i][1] + n_out[3],
                           crops[i][0] + n_out[4], crops[i][1] + n_out[5]]

                n_ori_imgs.append(ori_imgs[i])
                n_ori_crops.append(crops[i])
                n_r_crops.append(n_crops)

                n_img, n_lab = r_o_ge.next_test_R_O(ori_imgs[i], r_out[i], n_crops, crops[i])#img, lab, n_crops, crops
                n_imgs.append(n_img)
        if len(n_imgs) == 0 :
            continue
        o_out = sess.run(o_output,
                         feed_dict={img: n_imgs})

        for i in range(o_out.shape[0]):
            output_p = o_out[i, :]
            if output_p[0] > output_p[1]:
                img_show = n_ori_imgs[i]
                draw = ImageDraw.Draw(img_show)
                labeln = output_p[2:]
                labeln = [j*(n_r_crops[i][2]-n_r_crops[i][0])/48 for j in labeln]
                # im1 = Image.fromarray(deepcopy(n_img))
                # im1.show()
                # draw = ImageDraw.Draw(im1)
                # draw.rectangle(((n_lab[2], n_lab[3]), (n_lab[4], n_lab[5])), outline='red')
                # print(n_lab)
                # im1.show()
                # input()

                labeln = [k+n_r_crops[i][int(j % 2)] for j, k in enumerate(labeln)]

                draw.rectangle(((labeln[0], labeln[1]), (labeln[2], labeln[3])))
                draw.point((labeln[4], labeln[5]), fill='blue')
                draw.point((labeln[6], labeln[7]), fill='red')
                img_show.show()
                input()
        end_time = time.time()
        print('spend time: %.3f,' % (end_time - begin_time))

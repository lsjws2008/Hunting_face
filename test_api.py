import sys
sys.path.append("../Hunting_face")
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
# from generator import generator_test_region, generator_R_O_region

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def padding_img_rect(img):
    return padding_img(img, max(img.size))

def padding_img(img, size):
    new_im = Image.new("RGB", (size, size))
    new_im.paste(img, (0, 0))
    return new_im

def imgs_to_out(path):

    save_log = os.path.join('train_log', '3')

    save_folder = os.path.join(save_log, 'models')

    img_names = os.listdir(path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        r_net = R_Net.R_Net()

        o_net = O_Net.O_Net()

        img = tf.placeholder(tf.float32,
                             [None, None, None, 3])

        r_output = r_net.pr(img)

        o_output = o_net.pr(img)

        model_pr = tf.train.Saver()
        model_pr.restore(sess, os.path.join(save_folder,
                                            str(4000) + 'save_net.ckpt'))

        initialize_uninitialized(sess)

        imgs = [Image.open(os.path.join(path, i)) for i in img_names]
        imgs = [padding_img_rect(i) for i in imgs]
        p_imgs = [np.array(i.resize((24, 24), Image.BILINEAR)) for i in imgs]

        r_out = sess.run(r_output,
                         feed_dict={img: p_imgs})

        n_r_crops = []
        n_imgs = []
        n_ori_img = []

        for i in range(r_out.shape[0]):

            n_out = r_out[i]

            n_ori_img.append(imgs[i])
            n_crops = [n_out[2], n_out[3],
                       n_out[4], n_out[5]]
            n_r_crops.append(n_crops)

            im1 = Image.fromarray(p_imgs[i]).crop(n_crops)
            im1 = padding_img_rect(im1)
            im1 = im1.resize([48, 48], Image.BILINEAR)
            n_imgs.append(np.array(im1))

        o_out = sess.run(o_output,
                         feed_dict={img: n_imgs})

        for i in range(o_out.shape[0]):
            output_p = o_out[i, :]

            img_show = n_ori_img[i]
            draw = ImageDraw.Draw(img_show)
            labeln = output_p[2:]
            labeln = [j*(n_r_crops[i][2]-n_r_crops[i][0])/48 for j in labeln]
            labeln = [k+n_r_crops[i][int(j % 2)] for j, k in enumerate(labeln)]

            labeln = [j/24*max(img_show.size)for j in labeln]
            draw.rectangle(((labeln[0], labeln[1]), (labeln[2], labeln[3])), outline='black')
            draw.ellipse((labeln[4]-2, labeln[5]-2, labeln[4]+2, labeln[5]+2), fill='blue')
            draw.ellipse((labeln[6]-2, labeln[7]-2, labeln[6]+2, labeln[7]+2), fill='red')
            img_show.show()

if __name__ == '__main__':
    imgs_to_out('test_dir')

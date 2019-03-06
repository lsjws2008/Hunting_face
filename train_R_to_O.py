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
from generator import generator_img_region, generator_R_O_region
from copy import deepcopy


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def get_by_indx(imgs, labs, com_labs, loss_sorted):
    n_imgs = []
    n_labs = []
    n_com_labs = []
    for u in loss_sorted:
        n_imgs.append(imgs[u])
        n_labs.append(labs[u])
        n_com_labs.append(com_labs[u])
    return imgs, \
        labs, \
        com_labs


batch = 30
epoch = 1000000
lear = 1e-4
lss = []

parser = ArgumentParser()
parser.add_argument("-S", "--save-log", help="save to train_log", dest="save_log", default="3")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.4")

args = parser.parse_args()
save_log = os.path.join('train_log',
                        args.save_log)

r_save_log = os.path.join('train_log',
                          '1')
r_save_folder = os.path.join(r_save_log, 'models')

if not os.path.isdir(save_log):
    os.mkdir(save_log)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)

output_num = 10

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    r_net = R_Net.R_Net()
    r_loss_rate = [1, 0.5, 0.5]
    r_image_size = 24
    r_dy = [5, 11, 20]

    img = tf.placeholder(tf.float32,
                         [None, None, None, 3])

    r_output = r_net.pr(img)

    model_pr = tf.train.Saver()
    model_pr.restore(sess, os.path.join(r_save_folder,
                                        str(10000) + 'save_net.ckpt'))

    o_net = O_Net.O_Net()
    o_loss_rate = [1, 0.5, 1]
    o_image_size = 48
    o_dy = [10, 22, 40]

    ge = generator_img_region.generator(json_path='/data/train_label.json',
                                        image_path='/data/img_pyramids/',
                                        image_size=24,
                                        batch=int(10),
                                        image_from_each_face=3,
                                        dy=[5, 13, 20])

    r_o_ge = generator_R_O_region.generator_r_o()

    lab = tf.placeholder(tf.float32,
                         [None, output_num])
    com_lab = tf.placeholder(tf.float32,
                             [None, output_num])
    lr = tf.placeholder(tf.float32,
                        [None])

    o_output = o_net.pr(img)
    output = tf.reshape(o_output, [-1, 10])
    cls = output[:, :2]
    bbox = output[:, 2:6]
    eye_reg = output[:, 6:]

    lab_cls = lab[:, :2]
    lab_bbox = lab[:, 2:6]
    lab_eye_reg = lab[:, 6:]

    com_lab_cls = com_lab[:, :2]
    com_lab_bbox = com_lab[:, 2:6]
    com_lab_eye_reg = com_lab[:, 6:]

    cls_loss = losses.cross_entropy_loss(cls, lab_cls)
    bbox_loss = losses.Euclidean_loss(bbox, lab_bbox)*com_lab_bbox
    bbox_loss = tf.squeeze(bbox_loss)
    eye_reg_loss = losses.Euclidean_loss(eye_reg, lab_eye_reg)*com_lab_eye_reg
    eye_reg_loss = tf.squeeze(eye_reg_loss)

    loss = o_loss_rate[0] * cls_loss + \
           o_loss_rate[1] * tf.reduce_sum(bbox_loss, 1) + \
           o_loss_rate[2] * tf.reduce_sum(eye_reg_loss, 1)

    t_loss = loss

    train_step = tf.train.MomentumOptimizer(lr[0],
                                            0.9). \
        minimize(t_loss)

    model_af = tf.train.Saver()
    initialize_uninitialized(sess)

    print('begin: ')
    for seq in range(epoch):

        if (seq + 1) % 2000 == 0:
            lear *= 0.5
            print(lear)

        begin_time = time.time()

        n_imgs = []
        n_labs = []
        n_com_labs = []
        while len(n_imgs) < batch:
            imgs, \
            labs, \
            com_labs, \
            ori_imgs, \
            ori_labels, \
            crops \
                = ge.__next__()

            r_out = sess.run(r_output,
                             feed_dict={img: imgs
                                        })

            for i in range(r_out.shape[0]):
                if r_out[i][0] > r_out[i][1]:

                    n_out = r_out[i] * (crops[i][2] - crops[i][0])/24

                    n_crops = [crops[i][0]+n_out[2], crops[i][1]+n_out[3],
                               crops[i][0]+n_out[4], crops[i][1]+n_out[5]]

                    # draw = ImageDraw.Draw(ori_imgs[i])
                    # draw.rectangle(((crops[i][0], crops[i][1]), (crops[i][2], crops[i][3])), outline='black')
                    # draw.rectangle(((n_crops[0], n_crops[1]), (n_crops[2], n_crops[3])), outline='red')
                    # draw.rectangle(((ori_labels[i][1][0]+crops[i][0], ori_labels[i][1][1]+crops[i][1]), (ori_labels[i][1][2]+crops[i][0], ori_labels[i][1][3]+crops[i][1])), outline='green')
                    # ori_imgs[i].show()
                    n_img, n_lab, n_com_lab = r_o_ge.next_R_O(ori_imgs[i], ori_labels[i], n_crops, crops[i], com_labs[i])
                    # im1 = Image.fromarray(deepcopy(n_img))
                    # im1.show()
                    # draw = ImageDraw.Draw(im1)
                    # draw.rectangle(((n_lab[2], n_lab[3]), (n_lab[4], n_lab[5])), outline='red')
                    # print(n_lab)
                    # im1.show()
                    # input()
                    n_imgs.append(n_img)
                    n_labs.append(n_lab)
                    n_com_labs.append(n_com_lab)

        total_loss = sess.run(t_loss,
                 feed_dict={img: n_imgs,
                            lab: n_labs,
                            com_lab: n_com_labs,
                            lr: [lear]})

        loss_sorted = sorted(range(len(total_loss.tolist())), key=lambda k: total_loss[k])
        loss_sorted = loss_sorted[:int(len(total_loss) * 0.7)]

        n_imgs, \
        n_labs, \
        n_com_labs = \
        get_by_indx(n_imgs,
                    n_labs,
                    n_com_labs,
                    loss_sorted)

        sess.run(train_step,
                 feed_dict={img: n_imgs,
                            lab: n_labs,
                            com_lab: n_com_labs,
                            lr: [lear]})

        if seq % 5 == 0:

            print('\nSequence:', str(seq))

            [ls_t,
             out] = sess.run([loss,
                              output],
                             feed_dict={img: n_imgs,
                                        lab: n_labs,
                                        com_lab: n_com_labs,
                                        lr: [lear]})

            ls_t = np.sum(ls_t)
            lss.append(ls_t)
            if len(lss) > 1e4:
                lss.remove(lss[0])

            plt.plot(range(len(lss)), lss)
            feed_back_folder = os.path.join(save_log, 'feed_back')

            if not os.path.isdir(feed_back_folder):
                os.mkdir(feed_back_folder)
            plt.savefig(os.path.join(feed_back_folder,
                                     'l' + str(int(seq / 5e4)) + '.png'))
            #plt.show()
            plt.clf()

            if np.isnan(ls_t):
                input('isnan')

            avg_loss = sum([0.9 *\
                            math.pow(0.1,
                                     len(lss) - 1 - lsins)\
                            * ls\
                            for lsins, ls in enumerate(lss)])

            print('spand time: {0:.3f}, loss: {1:.3f}, max value: {2:.3f}'. \
                  format(time.time() - begin_time
                         , avg_loss,
                         np.amax(out)
                         ))

        if (seq + 1) % 1000 == 0:
            save_folder = os.path.join(save_log, 'models')

            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            model_af.save(sess,
                          os.path.join(save_folder,
                                       str(seq + 1) + 'save_net.ckpt'))

from argparse import ArgumentParser
import tensorflow as tf
from archs import P_Net, losses, O_Net, R_Net
import time
# from math import pow
import os
from random import shuffle
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
from generator import generator_img_region


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
def get_by_indx(imgs, labs, com_labs, ori_imgs, ori_labels, crops, loss_sorted):
    n_imgs = []
    n_labs = []
    n_com_labs = []
    n_ori_imgs = []
    n_ori_labels = []
    n_crops = []
    for u in loss_sorted:
        n_imgs.append(imgs[u])
        n_labs.append(labs[u])
        n_com_labs.append(com_labs[u])
        n_ori_imgs.append(ori_imgs[u])
        n_ori_labels.append(ori_labels[u])
        n_crops.append(crops[u])
    return imgs, \
        labs, \
        com_labs, \
        ori_imgs, \
        ori_labels, \
        crops


batch = 30
epoch = 1000000
lear = 1e-4
lss = []

parser = ArgumentParser()
parser.add_argument("-S", "--save-log", help="save to train_log", dest="save_log", default="3")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.4")
parser.add_argument("-P", "--part-train", help="which part to train ", type=str, dest="part_train", default="O")

args = parser.parse_args()


save_log = os.path.join('train_log',
                        args.save_log)

if not os.path.isdir(save_log):
    os.mkdir(save_log)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

output_num = 10

with tf.Session(config=config) as sess:
    if args.part_train == 'P':
        net = P_Net.P_Net()
        loss_rate = [1, 0.5, 0.5]
        image_size = 12
        dy = [5, 13, 20]

    elif args.part_train == 'R':
        net = R_Net.R_Net()
        loss_rate = [1, 0.5, 0.5]
        image_size = 24
        dy = [5, 13, 20]

    elif args.part_train == 'O':
        net = O_Net.O_Net()
        loss_rate = [1, 0.5, 1]
        image_size = 48
        dy = [10, 22, 40]

    ge = generator_img_region.generator(json_path='/data/train_label.json',
                                        image_path='/data/img_pyramids/',
                                        image_size=image_size,
                                        batch=int(batch / 3),
                                        image_from_each_face=3,
                                        dy=dy)

    img = tf.placeholder(tf.float32,
                         [None, None, None, 3])
    lab = tf.placeholder(tf.float32,
                         [None, output_num])
    com_lab = tf.placeholder(tf.float32,
                             [None, output_num])
    lr = tf.placeholder(tf.float32,
                        [None])

    n_output = net.pr(img)
    output = tf.reshape(n_output, [-1, 10])
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

    loss = loss_rate[0] * cls_loss + \
           loss_rate[1] * tf.reduce_sum(bbox_loss, 1) + \
           loss_rate[2] * tf.reduce_sum(eye_reg_loss, 1)
    t_loss = loss
    # loss = 0.5 * tf.reduce_sum(bbox_loss, 3) + 0.5 * tf.reduce_sum(eye_reg_loss, 3)

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

        imgs, \
        labs, \
        com_labs, \
        ori_imgs, \
        ori_labels, \
        crops \
            = ge.__next__(image_size)

        total_loss = sess.run(loss,
                              feed_dict={img: imgs,
                                         lab: labs,
                                         com_lab: com_labs,
                                         lr: [lear]})

        loss_sorted = sorted(range(len(total_loss.tolist())), key=lambda k: total_loss[k])
        loss_sorted = loss_sorted[:int(len(total_loss)*0.7)]

        imgs, \
        labs, \
        com_labs, \
        ori_imgs, \
        ori_labels, \
        crops = \
        get_by_indx(imgs,
                    labs,
                    com_labs,
                    ori_imgs,
                    ori_labels,
                    crops,
                    loss_sorted)

        sess.run(train_step,
                 feed_dict={img: imgs,
                            lab: labs,
                            com_lab: com_labs,
                            lr: [lear]})

        if seq % 5 == 0:

            print('\nSequence:', str(seq))

            [ls_t,
             out] = sess.run([t_loss,
                              output],
                             feed_dict={img: imgs,
                                        lab: labs,
                                        com_lab: com_labs,
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

            avg_loss = sum([0.9 * \
                            math.pow(0.1,
                                     len(lss) - 1 - lsins) \
                            * ls \
                            for lsins, ls in enumerate(lss)])

            print('spand time: {0:.3f}, loss: {1:.3f}'. \
                  format(time.time() - begin_time
                         , avg_loss
                         ))

        if (seq + 1) % 1000 == 0:
            save_folder = os.path.join(save_log, 'models')

            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            model_af.save(sess,
                          os.path.join(save_folder,
                                       str(seq + 1) + 'save_net.ckpt'))

from argparse import ArgumentParser
import tensorflow as tf
from archs import R_Net, losses
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


batch = 30
epoch = 10000
lear = 1e-4
lss = []
image_size = 24
ge = generator_img_region.generator(json_path='/data/train_label.json',
                                    image_path='/data/img_pyramids/',
                                    image_size=image_size,
                                    batch=int(batch/6),
                                    image_from_each_face=6,
                                    dy=[5, 13, 20])

parser = ArgumentParser()
parser.add_argument("-s", "--save-log", help="save to train_log", dest="save_log", default="1")
parser.add_argument("-G", "--gpu-memory", help="gpu memary used", type=float, dest="gpu_memory", default="0.4")
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
    net = R_Net.R_Net()

    img = tf.placeholder(tf.float32,
                         [batch, None, None, 3])
    lab = tf.placeholder(tf.float32,
                         [batch, output_num])
    com_lab = tf.placeholder(tf.float32,
                             [batch, output_num])
    lr = tf.placeholder(tf.float32,
                        [None])

    output = net.pr(img)
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

    loss = 1 * tf.reduce_sum(cls_loss) + 0.5 * tf.reduce_sum(bbox_loss) + 0.5 * tf.reduce_sum(eye_reg_loss)
    # loss = 0.5 * tf.reduce_sum(bbox_loss, 3) + 0.5 * tf.reduce_sum(eye_reg_loss, 3)

    train_step = tf.train.MomentumOptimizer(lr[0],
                                            0.9). \
        minimize(loss)

    model_af = tf.train.Saver()
    initialize_uninitialized(sess)

    print('begin', end=' :')
    for seq in range(epoch):

        if (seq + 1) % 1000 == 0:
            lear *= 0.5
            print(lear)

        begin_time = time.time()

        imgs, \
        labs, com_labs\
            = ge.__next__(image_size)

        sess.run(train_step,
                 feed_dict={img: imgs,
                            lab: labs,
                            com_lab: com_labs,
                            lr: [lear]})

        if seq % 5 == 0:

            print('\nSequence:', str(seq))

            [ls_t,
             out] = sess.run([loss,
                              output],
                             feed_dict={img: imgs,
                                        lab: labs,
                                        com_lab: com_labs,
                                        lr: [lear]})
            out = np.squeeze(out)
            print(np.amax(labs))
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

            avg_loss = sum([0.9 * \
                            math.pow(0.1,
                                     len(lss) - 1 - lsins) \
                            * ls \
                            for lsins, ls in enumerate(lss)])

            print('spand time: {0:.3f}, loss: {1:.3f}, max value: {2:.3f}'. \
                  format(time.time() - begin_time
                         , avg_loss,
                         np.amax(out[:, 2:])
                         ))
            # print(out)

        if (seq + 1) % 1000 == 0:
            save_folder = os.path.join(save_log, 'models')

            if not os.path.isdir(save_folder):
                os.mkdir(save_folder)

            model_af.save(sess,
                          os.path.join(save_folder,
                                       str(seq + 1) + 'save_net.ckpt'))

import numpy as np
from PIL import Image, ImageDraw
import json
import random
from copy import deepcopy
from datetime import datetime
from math import ceil, log
from generator.generator_img_region import generator


class generator_r_o():

    # def next_R(self, size=0):
    #
    #     if size != 0:
    #         self.size = size
    #
    #     if len(self.image_indx) < self.batch*2:
    #         self.init_img_indx()
    #
    #     images = []
    #     labels = []
    #     for i in range(int(self.batch)):
    #         image = self.read_img(self.path, self.image_names[self.image_indx[0]])
    #         label = self.image_labels[self.image_indx[0]]
    #         data = self.resize_img_by_scale(image, [1, 1], *label, 224 / max(image.size))
    #         image, label = self.crop_controler(data[0], list(data[2:]))
    #         for j, k in zip(image, label):
    #             image = self.padding_img_rect(j)
    #             data = self.resize_img_by_scale(image, *k, self.size / max(image.size)) ###
    #             image = self.resize_img(image, self.size, self.size)
    #
    #             images.append(image)
    #             labels.append(data[1:])
    #
    #             # draw = ImageDraw.Draw(image)
    #             # labeln = data[1:]
    #             # draw.rectangle(((labeln[1][0], labeln[1][1]), (labeln[1][2]+labeln[1][0], labeln[1][3]+labeln[1][1])))
    #             # draw.point((labeln[2][0],labeln[2][1]),fill='red')
    #             # draw.point((labeln[3][0], labeln[3][1]),fill='red')
    #             # image.show()
    #             # input(labeln[1])
    #         self.image_indx.remove(self.image_indx[0])
    #
    #     images, labels = self.shuffle_image_label(images, labels)
    #
    #     n_labels = []
    #
    #     for i in labels:
    #         n_label = [k for j in i for k in j]
    #         n_label = np.array(n_label)
    #         n_label[n_label < 0] = 0
    #         # if n_label[0] == 1:
    #         #     n_label_reg = self.bbox_to_bboxreg([0, 0, self.size, self.size], n_label[2:6])
    #         #     n_label = np.concatenate([n_label[:2], n_label_reg, n_label[6:]])
    #         n_labels.append(n_label)
    #
    #     com_labels = deepcopy(n_labels)
    #     for i in com_labels:
    #         if i[0] == 0:
    #             i[2:] = [0]*(len(i)-2)
    #             i[0] = 1
    #         if i[0] == 0.5:
    #             i[2:] = [1] * (len(i) - 2)
    #             i[0] = 1
    #             i[1] = 1
    #         if i[0] == 1:
    #             i[2:] = [1]*(len(i)-2)
    #             i[1] = 1
    #
    #     images = [np.array(i) for i in images]
    #     # print(n_labels[0])
    #     return images, n_labels, com_labels

    def next_R_O(self, img, lab, n_crops, crops, com_lab):
        # img = Image.fromarray(img)
        img = deepcopy(img)
        img = img.crop(n_crops)
        lab = np.concatenate(deepcopy(lab), 0)
        lab = np.concatenate([lab[:2], [j-n_crops[int(i % 2)]+crops[int(i % 2)] for i, j in enumerate(lab[2:])]], 0)
        img = self.padding_img_rect(img)
        size = img.size
        img = self.resize_img(img, 48, 48)
        lab[2:] = lab[2:] * (48 / max(size))
        lab[lab<0] = 0
        lab[lab>48] = 48
        return np.array(img), lab, com_lab

    def next_test_R_O(self, img, lab, n_crops, crops):
        # img = Image.fromarray(img)
        img = deepcopy(img)
        img = img.crop(n_crops)
        lab = deepcopy(lab)
        lab = np.concatenate([lab[:2], [j-n_crops[int(i % 2)]+crops[int(i % 2)] for i, j in enumerate(lab[2:])]], 0)
        img = self.padding_img_rect(img)
        size = img.size
        img = self.resize_img(img, 48, 48)
        lab[2:] = lab[2:] * (48 / max(size))
        lab[lab<0] = 0
        lab[lab>48] = 48
        return np.array(img), lab

    def padding_img_rect(self, img):
        return self.padding_img(img, max(img.size))

    def padding_img(self, img, size):
        new_im = Image.new("RGB", (size, size))
        new_im.paste(img, (0, 0))
        return new_im

    def resize_label_by_scale(self, img_mask, label_bbox, label_eye1, label_eye2, scale):
        return img_mask, \
               [i * scale for i in label_bbox], \
               [i * scale for i in label_eye1], \
               [i * scale for i in label_eye2]

    def resize_img(self, img, width, height):
        return img.resize((width, height), Image.BILINEAR)

    def resize_img_by_scale(self, img, img_mask, label_bbox, label_eye1, label_eye2, scale):
        return self.resize_img(img, ceil(img.size[0]*scale), ceil(img.size[1]*scale)),\
               img_mask, \
               [i * scale for i in label_bbox], \
               [i * scale for i in label_eye1], \
               [i * scale for i in label_eye2]


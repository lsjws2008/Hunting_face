import numpy as np
from PIL import Image, ImageDraw
import json
from random import shuffle
import random
from copy import deepcopy
from generator import generator_img_region
import scipy.io


class generator_test(generator_img_region.generator):

    def __init__(self, image_path, image_size, batch, image_from_each_face, dy):
        mat = scipy.io.loadmat('/data/LabelTestAll.mat')
        self.label = mat.get('LabelTest')
        self.generator_group()
        self.size = image_size
        self.init_img_indx()
        self.batch = batch
        self.path = image_path
        self.face = image_from_each_face
        self.dy = dy

    def __next__(self, size=0):

        if size != 0:
            self.size = size

        if len(self.image_indx) < self.batch * 2:
            self.init_img_indx()

        images = []
        labels = []
        ori_img = []
        crops_1 = []

        for i in range(int(self.batch)):
            image = self.read_img(self.path, self.image_names[self.image_indx[0]])
            label = self.image_labels[self.image_indx[0]]
            data = self.resize_img_by_scale(image, [1, 1], label[0], 224 / max(image.size))
            image_1, label_1, crops = self.crop_controler(data[0], list(data[2:]), istrain=False)
            for j, k, t in zip(image_1, label_1, crops):
                image = self.padding_img_rect(j)
                # image.show()
                ori_img.append(deepcopy(data[0]))
                crops_1.append(t)
                data_1 = self.resize_img_by_scale(image, *k, self.size / max(image.size))
                image = self.resize_img(image, self.size, self.size)

                images.append(image)
                labels.append(data_1[1:])

                # draw = ImageDraw.Draw(data[0])
                # labeln = data[1:]
                # draw.rectangle(((labeln[1][0], labeln[1][1]), (labeln[1][2]+labeln[1][0], labeln[1][3]+labeln[1][1])))
                # data[0].show()

            self.image_indx.remove(self.image_indx[0])

        #images, labels = self.shuffle_image_label(images, labels)
        images = [np.array(i) for i in images]

        return images, labels, ori_img, crops_1


    def resize_img_by_scale(self, img, img_mask, label_bbox, scale):
        return self.resize_img(img, int(img.size[0]*scale), int(img.size[1]*scale)),\
               img_mask, \
               [i * scale for i in label_bbox]

    def generator_group(self):
        num_data = self.label.shape[1]
        self.image_names = []
        self.image_labels = []
        for i in range(num_data):
            img_name = self.label[0][i][0][0]
            label = [self.label[0][i][1][0][:4]]
            self.image_names.append(img_name)
            self.image_labels.append(label)


if __name__ == '__main__':
    ge = generator_test(image_path='/data/images/',
                        image_size=12,
                        batch=1,
                        image_from_each_face=2)
    ge.__next__()

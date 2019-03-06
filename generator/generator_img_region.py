import numpy as np
from PIL import Image, ImageDraw
import json
import random
from copy import deepcopy
from datetime import datetime
from math import ceil, log


class generator():

    def __init__(self, json_path, image_path, image_size, batch, image_from_each_face, dy):
        with open(json_path, encoding='utf-8') as f:
            data = f.read()
        data = json.loads(data)
        self.image_names = list(data.keys())
        self.image_labels = [data[i] for i in self.image_names]
        self.size = image_size
        self.init_img_indx()
        self.batch = batch
        self.path = image_path
        self.face = image_from_each_face
        self.dy = dy

    def init_img_indx(self):
        random.seed(datetime.now())
        self.image_indx = list(range(len(self.image_names)))
        random.shuffle(self.image_indx)

    def __next__(self, size=0):

        if size != 0:
            self.size = size

        if len(self.image_indx) < self.batch*2:
            self.init_img_indx()

        images = []
        labels = []
        crops = []
        ori_img = []
        ori_label = []
        for i in range(int(self.batch)):
            image = self.read_img(self.path, self.image_names[self.image_indx[0]])
            label = self.image_labels[self.image_indx[0]]
            data = self.resize_img_by_scale(image, [1, 1], *label, 224 / max(image.size))

            image, label, crop = self.crop_controler(data[0], list(data[2:]))

            for j, k, t in zip(image, label, crop):
                image = self.padding_img_rect(j)

                crops.append(t)
                ori_img.append(deepcopy(data[0]))
                ori_label.append(k)
                if max(image.size) == 0:
                    data_1 = self.resize_img_by_scale(image, *k, self.size / 224) ###
                else:
                    data_1 = self.resize_img_by_scale(image, *k, self.size / max(image.size)) ###

                image = self.resize_img(image, self.size, self.size)
                images.append(image)
                labels.append(data_1[1:])

                # draw = ImageDraw.Draw(image)
                # labeln = data[1:]
                # draw.rectangle(((labeln[1][0], labeln[1][1]), (labeln[1][2]+labeln[1][0], labeln[1][3]+labeln[1][1])))
                # draw.point((labeln[2][0],labeln[2][1]),fill='red')
                # draw.point((labeln[3][0], labeln[3][1]),fill='red')
                # image.show()
                # input(labeln[1])
            self.image_indx.remove(self.image_indx[0])

        images, labels, ori_img, ori_label, crops = self.shuffle_image_label(images, labels, ori_img, ori_label, crops)

        n_labels = []

        for i in labels:
            n_label = [k for j in i for k in j]
            n_label = np.array(n_label)
            n_label[n_label < 0] = 0
            # if n_label[0] == 1:
                # n_label_reg = self.bbox_to_bboxreg([0, 0, self.size, self.size], n_label[2:6])
                # n_label = np.concatenate([n_label[:2], n_label_reg, n_label[6:]])
            n_label[n_label > self.size] = self.size
            n_labels.append(n_label)

        com_labels = deepcopy(n_labels)
        for i in com_labels:
            if i[0] == 0:
                i[2:] = [0]*(len(i)-2)
                i[0] = 1
            if i[0] == 0.5:
                i[2:] = [1] * (len(i) - 2)
                i[0] = 1
                i[1] = 1
            if i[0] == 1:
                i[2:] = [1]*(len(i)-2)
                i[1] = 1

        images = [np.array(i) for i in images]
        # print(n_labels[0])
        return images, n_labels, com_labels, ori_img, ori_label, crops

    def shuffle_image_label(self, images, labels, ori_img, ori_label, crops):
        random.seed(datetime.now())
        image_indx = list(range(len(images)))
        random.shuffle(image_indx)
        n_images = []
        n_labels = []
        n_ori_img = []
        n_ori_label = []
        n_crops = []
        for i in image_indx:
            n_images.append(images[i])
            n_labels.append(labels[i])
            n_ori_img.append(ori_img[i])
            n_ori_label.append(ori_label[i])
            n_crops.append(crops[i])
        return  n_images, n_labels, n_ori_img, n_ori_label, n_crops

    def bbox_to_bboxreg(self, pbox, gbox):
        dx = (gbox[0]-pbox[0])/pbox[2]
        dy = (gbox[1]-pbox[1])/pbox[3]
        dw = log(gbox[2]/pbox[2])
        dh = log(gbox[3]/pbox[3])

        return [dx, dy, dw, dh]

    def crop_controler(self, im, label, istrain=True):
        part_img = []
        part_label = []
        part_crops = []
        posi_img = []
        posi_label = []
        posi_crops = []
        nega_img = []
        nega_label = []
        nega_crops = []

        while not len(posi_img) == int(self.face/3):
            box = label[0]
            box = (box[0], box[1], box[0]+box[2], box[1]+box[3])
            crop_area = self.get_crop_area(self.dy[0])
            crop_area = [i+j for i, j in zip(box, crop_area)]

            IoU = self.bb_intersection_over_union(crop_area, box)
            # input('1'+str(IoU))
            if IoU > 0.65:
                # print('IoU2', IoU)
                posi_crops.append(crop_area)
                imp, labelp = self.crop_img_label(crop_area, im, label, istrain=istrain)

                # draw = ImageDraw.Draw(imp)
                # draw.rectangle(((labelp[0][0], labelp[0][1]), (labelp[0][2]+labelp[0][0], labelp[0][3]+labelp[0][1])))
                # draw.point((labelp[1][0],labelp[1][1]),fill='red')
                # draw.point((labelp[2][0], labelp[2][1]),fill='red')
                # imp.show()
                labelp.insert(0, [1, 0])
                posi_img.append(imp)
                posi_label.append(labelp)

        while not len(part_img) == int(self.face/3):
            box = label[0]
            box = (box[0], box[1], box[0]+box[2], box[1]+box[3])
            crop_area = self.get_crop_area(self.dy[1])
            crop_area = [i+j for i, j in zip(box, crop_area)]
            IoU = self.bb_intersection_over_union(crop_area, box)
            if IoU < 0.65 and IoU >= 0.4 :
                # print('IoU2', IoU)
                part_crops.append(crop_area)
                imp, labelp = self.crop_img_label(crop_area, im, label, istrain=istrain)

                # draw = ImageDraw.Draw(imp)
                # draw.rectangle(((labelp[0][0], labelp[0][1]), (labelp[0][2]+labelp[0][0], labelp[0][3]+labelp[0][1])))
                # draw.point((labelp[1][0],labelp[1][1]),fill='red')
                # draw.point((labelp[2][0], labelp[2][1]),fill='red')
                # imp.show()
                labelp.insert(0, [0, 1])
                part_img.append(imp)
                part_label.append(labelp)

        while not len(nega_img) == int(self.face / 3):
            box = label[0]
            box = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            crop_area = self.get_crop_area(self.dy[2])
            bbox = [box[2], box[3]]
            crop_area = [bbox[0] + crop_area[0], bbox[1] + crop_area[1], bbox[0] + crop_area[2], bbox[1] + crop_area[3]]

            IoU = self.bb_intersection_over_union(crop_area, box)
            if IoU < 0.35:
                nega_crops.append(crop_area)
                imn, labeln = self.crop_img_label(crop_area, im, label, istrain=istrain)

                # draw = ImageDraw.Draw(imn)
                # draw.rectangle(((labeln[0][0], labeln[0][1]), (labeln[0][2]+labeln[0][0], labeln[0][3]+labeln[0][1])))
                # draw.point((labeln[1][0],labeln[1][1]),fill='red')
                # draw.point((labeln[2][0], labeln[2][1]),fill='red')
                # imn.show()

                labeln.insert(0, [0, 1])
                nega_img.append(imn)
                nega_label.append(labeln)
        return posi_img+part_img+nega_img, posi_label+part_label + nega_label, posi_crops+part_crops+nega_crops

    def crop_img_label(self, crop_area, im, label, istrain=True):
        im = im.crop(crop_area)

        lp = deepcopy(label)

        lp[0][0] = lp[0][0] - crop_area[0]
        lp[0][1] = lp[0][1] - crop_area[1]
        if istrain:
            lp[1][0] = lp[1][0] - crop_area[0]
            lp[1][1] = lp[1][1] - crop_area[1]

            lp[2][0] = lp[2][0] - crop_area[0]
            lp[2][1] = lp[2][1] - crop_area[1]
        return im, lp

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if(boxAArea + boxBArea - interArea == 0):
            iou = 0
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def get_crop_area(self, dy):
        random.seed(datetime.now())
        crop_area = [random.randint(-dy*2, 0) for i in range(2)]
        [crop_area.append(random.randint(0, dy*2)) for i in range(2)]
        return crop_area

    def padding_img_rect(self, img):
        return self.padding_img(img, max(img.size))

    def padding_img(self, img, size):
        new_im = Image.new("RGB", (size, size))
        new_im.paste(img, (0, 0))
        return new_im

    def resize_img_by_scale(self, img, img_mask, label_bbox, label_eye1, label_eye2, scale):
        return self.resize_img(img, ceil(img.size[0]*scale), ceil(img.size[1]*scale)),\
               img_mask, \
               [i * scale for i in label_bbox], \
               [i * scale for i in label_eye1], \
               [i * scale for i in label_eye2]

    def resize_label_by_scale(self, img_mask, label_bbox, label_eye1, label_eye2, scale):
        return img_mask, \
               [i * scale for i in label_bbox], \
               [i * scale for i in label_eye1], \
               [i * scale for i in label_eye2]

    def read_img(self, path, img_name):
        return Image.open(path + img_name)

    def resize_img(self, img, width, height):
        return img.resize((width, height), Image.BILINEAR)


if __name__ == '__main__':
    ge = generator(json_path='/data/train_label.json',
                   image_path='/data/img_pyramids/',
                   image_size=56,
                   batch=8,
                   image_from_each_face=4)
    for i in range(10000):

        print(i)
        ge.__next__(24)

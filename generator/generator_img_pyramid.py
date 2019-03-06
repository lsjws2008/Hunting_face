import scipy.io
import numpy as np
from PIL import Image, ImageDraw
import json



#image_name = label[0][a][1][0]
#image_label = label[0][a][2][0]
#(x,y,w,h, x1,y1,x2,y2, x3,y3,w3,h3, occ_type, occ_degree, gender, race, orientation, x4,y4,w4,h4),
#(a) (x,y,w,h) is the bounding box of a face,
#(b) (x1,y1,x2,y2) is the position of two eyes.
# draw = ImageDraw.Draw(img)
# draw.rectangle(((label[0], label[1]), (label[0]+label[2], label[1]+label[3])))
# draw.point((label[4],label[5]),fill='red')
# draw.point((label[6], label[7]),fill='red')

class generator_img_pyramid():

    def __init__(self, min_img_rate, num_pyramid_layer, path='/data/images/', save_path='/data/img_pyramids/'):
        mat = scipy.io.loadmat('/data/LabelTrainAll.mat')
        self.label = mat.get('label_train')
        self.min_img_rate = min_img_rate
        self.num_pyramid_layer = num_pyramid_layer
        self.path = path
        self.save_path = save_path

    def generator_group(self):
        num_data = self.label.shape[1]
        data = {}

        for i in range(num_data):
            img_name = self.label[0][i][1][0]

            print(str(i)+' '+img_name)

            img = self.read_img(self.path, img_name)
            label = self.read_img_anno(self.label[0][i][2][0])
            labels = self.get_img_pyramid(img, *label, self.min_img_rate, self.num_pyramid_layer)
            imgs = [i[0] for i in labels]
            img_names = self.create_img_names(img_name)
            self.save_img_group(img_names, imgs)
            labels = {i:j[1:] for i, j in zip(img_names, labels)}
            data.update(labels)

        with open('/data/train_label.json', 'w') as outfile:
            json.dump(data, outfile)

    def create_img_names(self, img_name):
        img_name_pr = img_name.split('.')[0]
        img_name_end = img_name.split('.')[1]
        return [img_name_pr + '_' + str(i).zfill(3)+'.'+img_name_end for i in range(self.num_pyramid_layer)]

    def save_img_group(self, img_names,img_group):
        [self.save_img(i, j) for i, j in zip(img_names, img_group)]

    def save_img(self, img_name, img):
        img = img.convert('RGB')
        img.save(self.save_path + img_name)

    def get_img_pyramid(self, img, label_bbox, label_eye1, label_eye2, min_img_rate, num_pyramid_layer):
        scales = [(1-min_img_rate)/num_pyramid_layer*i + min_img_rate for i in range(num_pyramid_layer)]
        return [self.resize_img_by_scale(img, label_bbox, label_eye1, label_eye2, i) for i in scales]

    def resize_img_by_scale(self, img, label_bbox, label_eye1, label_eye2, scale):
        return self.resize_img(img, *(int(i*scale) for i in img.size)), \
               [i*scale for i in label_bbox], \
               [i*scale for i in label_eye1], \
               [i*scale for i in label_eye2]

    def resize_img(self, img, width, height):
        return img.resize((width, height), Image.BILINEAR)

    def read_img_anno(self, data):
        return data[:4], data[4:6], data[6:8]

    def read_img(self, path, img_name):
        return Image.open(path + img_name)


if __name__ == '__main__':
    ge = generator_img_pyramid(0.5, 5)
    ge.generator_group()

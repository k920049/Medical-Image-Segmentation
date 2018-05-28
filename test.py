import tensorflow as tf

A = tf.random_normal(shape=(16, 2))
soft_target = tf.nn.softmax(A)
soft_target = tf.arg_max(soft_target, 1)
soft_target = tf.expand_dims(soft_target, -1)
print(soft_target.shape)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n = sess.run([soft_target])
    print(n)

"""
from __future__ import print_function

import os

import caffe
import numpy as np
import lmdb
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from caffe.proto import caffe_pb2

lmdb_file = "/Users/jeasungpark/Plugins/DIGITS/digits/jobs/20180524-142734-6364/train_db/labels"
save_path = "./data/sunnybrook/temp"

with lmdb.open(lmdb_file) as lmdb_env:
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        data = np.clip(data, 0, 255)
        data = data.astype(np.uint8)
        for i in range(len(data)):
            image = data[i, :, :]
            if np.max(image) == 1:
                image = image * 255
            image = Image.fromarray(image, mode="L")
            path = str(key) + "_channel_" + str(i) + ".jpeg"
            image.save(os.path.join(save_path, path), format="jpeg")



from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot, cm

import os
import Queue
import collections

from src.load_data import get_all_contours, load_contour, shrink_case
from src.DataTuple import DataTuple

image_path = "./data/sunnybrook/challenge_training"
contour_path = "./data/sunnybrook/Sunnybrook Cardiac MR Database ContoursPart3/TrainingDataContours"

_ElementTuple = collections.namedtuple("ElementTuple", ("pixel", "depth"))


class ElementTuple(_ElementTuple):

    __slots__ = ()

    @property
    def dtype(self):
        (pixel, depth) = self

        if pixel.dtype == depth.dtype:
            return pixel.dtype
        else:
            raise TypeError("Inconsistent internal state: %s vs %s" % (str(pixel.dtype), str(depth.dtype)))


def bfs(label):
    new_label = np.zeros(shape=label.shape, dtype=np.int32)
    direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def within_range(pix):
        if pix[0] >= 0 and pix[0] < label.shape[0] and pix[1] >= 0 and pix[1] < label.shape[1]:
            return True
        return False

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            # Only search elements in the border
            if label[i, j] == 255:
                print("Processing ({}, {})".format(i, j))
                visited = np.zeros(shape=label.shape, dtype=np.int32)
                q = Queue.Queue()
                # Do bfs
                q.put_nowait(ElementTuple([i, j], depth=0))
                visited[i, j] = 1
                while not q.empty():
                    cur = q.get_nowait()
                    assert isinstance(cur, ElementTuple), "Error: The current element is not an instance of ElemTuple"

                    if new_label[cur.pixel[0], cur.pixel[1]] != 0:
                        if new_label[cur.pixel[0], cur.pixel[1]] > cur.depth:
                            new_label[cur.pixel[0], cur.pixel[1]] = cur.depth
                        else:
                            continue

                    if label[cur.pixel[0], cur.pixel[1]] == 0:
                        new_label[cur.pixel[0], cur.pixel[1]] = cur.depth

                    for d in direction:
                        next_element = [cur.pixel[0] + d[0], cur.pixel[1] + d[1]]
                        if within_range(next_element) and visited[next_element[0], next_element[1]] == 0:
                            visited[next_element[0], next_element[1]] = 1
                            q.put_nowait(ElementTuple(next_element, cur.depth + 1))

    return new_label

def Main():

    directories = [x[0] for x in os.walk(image_path)]
    case = []

    for dir in directories:
        print(dir)
        token = dir.split("/")
        if token[-1] != "challenge_training":
            case.append(token[-1])

    var_dict = {}

    shape = None

    contours = get_all_contours(contour_path)
    for elem in contours:
        image, label = load_contour(elem, image_path)

        if elem.case not in var_dict:
            var_dict[elem.case] = []

        var_dict[elem.case].append(DataTuple(image, label))
    # Image augmentation
    print("The number of images before augmentation: {}".format(len(var_dict[case[0]])))

    shown = False
    angles = [45, 90, 135, 180, 225, 270, 315]
    for c in case:
        bags = []
        for pair in var_dict[c]:
            assert isinstance(pair, DataTuple), "Error: Datatype mismatches"
            rows, cols = pair.image.shape
            size = rows * cols
            # Rotation
            for a in angles:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), a, 1)
                image = cv2.warpAffine(pair.image, M, (cols, rows))
                label = cv2.warpAffine(pair.label, M, (cols, rows))
                bags.append(DataTuple(image, label))
            # Blur
            image = cv2.GaussianBlur(pair.image, (5, 5), 0)
            bags.append(DataTuple(image, pair.label))
            # Flip
            image_x = cv2.flip(pair.image, flipCode=0)
            label_x = cv2.flip(pair.label, flipCode=0)
            image_y = cv2.flip(pair.image, flipCode=1)
            label_y = cv2.flip(pair.label, flipCode=1)
            bags.append(DataTuple(image_x, label_x))
            bags.append(DataTuple(image_y, label_y))
            # Add noise(5%)
            noise = np.random.randint(-50, 50, pair.image.shape)
            image = pair.image + noise
            bags.append(DataTuple(image, pair.label))

            if not shown:
                fig, axes = pyplot.subplots(ncols=3, figsize=(11, 6))
                label = pair.label
                # label = cv2.Canny(label, threshold1=0, threshold2=1)
                # label = bfs(label)

                new_label = np.empty(shape=(label.shape[0], label.shape[1], 2), dtype=np.int32)

                for i in range(label.shape[0]):
                    for j in range(label.shape[1]):
                        if label[i, j] == 0:
                            new_label[i, j, 0] = 1
                            new_label[i, j, 1] = 0
                        else:
                            new_label[i, j, 0] = 0
                            new_label[i, j, 1] = 1

                axes[0].imshow(new_label[:, :, 0],
                               cmap=cm.gray,
                               aspect="equal",
                               interpolation="none",
                               vmin=0, vmax=1)
                axes[1].imshow(new_label[:, :, 1],
                               cmap=cm.gray,
                               aspect="equal",
                               interpolation="none",
                               vmin=0, vmax=1)
                pyplot.show()
                shown = True

        for elem in bags:
            var_dict[c].append(elem)

    print("The number of images after augmentation: {}".format(len(var_dict[case[0]])))



if __name__ == "__main__":
    Main()
"""
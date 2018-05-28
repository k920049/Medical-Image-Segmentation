# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import, print_function

import cv2
import fnmatch
import math
import os
import random
import re
import Queue
import collections

import pydicom
import numpy as np

from digits.utils import subclass, override, constants
from digits.utils.constants import COLOR_PALETTE_ATTRIBUTE
from digits.extensions.data.interface import DataIngestionInterface
from .forms import DatasetForm, InferenceForm


DATASET_TEMPLATE = "template/dataset_template.html"
INFERENCE_TEMPLATE = "template/inference_template.html"


# This is the subset of SAX series to use for Left Ventricle segmentation
# in the challenge training dataset
SAX_SERIES = {

    "SC-HF-I-01": "0004",
    "SC-HF-I-02": "0106",
    "SC-HF-I-04": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-03": "0379",
    "SC-HF-NI-04": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-01": "0550",
    "SC-HYP-03": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-02": "0898",
    "SC-N-03": "0915",
    "SC-N-40": "0944",
}

#
# Utility functions
#


def shrink_case(case):
    toks = case.split("-")

    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])


class Contour(object):

    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = match.group(1)
        self.img_no = int(match.group(2))

    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)

    __repr__ = __str__


def get_all_contours(contour_path):
    # walk the directory structure for all the contour files
    contours = [
        os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')
    ]
    extracted = map(Contour, contours)
    return extracted


def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    img = load_image(full_path)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    return img, label


def load_image(full_path):
    f = pydicom.dcmread(full_path)
    return f.pixel_array.astype(np.int)



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


@subclass
class DataIngestion(DataIngestionInterface):
    """
    A data ingestion extension for the Sunnybrook dataset
    """

    def __init__(self, is_inference_db=False, **kwargs):
        super(DataIngestion, self).__init__(**kwargs)

        self.userdata['is_inference_db'] = is_inference_db

        self.userdata['class_labels'] = ['background', 'left ventricle']

        # get list of contours
        if 'contours' not in self.userdata:
            contours = get_all_contours(self.contour_folder)
            random.shuffle(contours)
            self.userdata['contours'] = contours
        else:
            contours = self.userdata['contours']

        # get number of validation entries
        pct_val = int(self.folder_pct_val)
        self.userdata['n_val_entries'] = int(math.floor(len(contours) * pct_val / 100))

        # label palette (0->black (background), 1->white (foreground), others->black)
        palette = [0, 0, 0,  255, 255, 255] + [0] * (254 * 3)
        self.userdata[COLOR_PALETTE_ATTRIBUTE] = palette

    @override
    def encode_entry(self, entry):
        if isinstance(entry, basestring):
            img = load_image(entry)
            label = np.array([0])
        else:
            img, label = load_contour(entry, self.image_folder)
            # label = label[np.newaxis, ...]
        # processing images
        if self.userdata['channel_conversion'] == 'L':
            feature = img[np.newaxis, ...]
        elif self.userdata['channel_conversion'] == 'RGB':
            feature = np.empty(shape=(3, img.shape[0], img.shape[1]), dtype=img.dtype)
            # just copy the same data over the three color channels
            feature[0] = img
            feature[1] = img
            feature[2] = img
        # processing labels
        new_label = np.empty(shape=(2, label.shape[0], label.shape[1]),
                             dtype=np.int32)

        label = label.astype(np.uint8)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i, j] == 0:
                    new_label[0, i, j] = 1
                    new_label[1, i, j] = 0
                else:
                    new_label[0, i, j] = 0
                    new_label[1, i, j] = 1

        label = cv2.Canny(label, threshold1=0, threshold2=1)
        label = self.bfs(label)
        label = label[np.newaxis, ...]
        label = np.concatenate([new_label, label], axis=0)

        return feature, label

    @staticmethod
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

    @staticmethod
    @override
    def get_category():
        return "Images"

    @staticmethod
    @override
    def get_id():
        return "image-sunnybrook-cardiac"

    @staticmethod
    @override
    def get_dataset_form():
        return DatasetForm()

    @staticmethod
    @override
    def get_dataset_template(form):
        """
        parameters:
        - form: form returned by get_dataset_form(). This may be populated
           with values if the job was cloned
        return:
        - (template, context) tuple
          - template is a Jinja template to use for rendering dataset creation
          options
          - context is a dictionary of context variables to use for rendering
          the form
        """
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, DATASET_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @override
    def get_inference_form(self):
        n_val_entries = self.userdata['n_val_entries']
        form = InferenceForm()
        for idx, ctr in enumerate(self.userdata['contours'][:n_val_entries]):
            form.validation_record.choices.append((str(idx), ctr.case))
        return form

    @staticmethod
    @override
    def get_inference_template(form):
        extension_dir = os.path.dirname(os.path.abspath(__file__))
        template = open(os.path.join(extension_dir, INFERENCE_TEMPLATE), "r").read()
        context = {'form': form}
        return (template, context)

    @staticmethod
    @override
    def get_title():
        return "UNet Sunnybrook LV Segmentation"

    @override
    def itemize_entries(self, stage):
        ctrs = self.userdata['contours']
        n_val_entries = self.userdata['n_val_entries']

        entries = []
        if not self.userdata['is_inference_db']:
            if stage == constants.TRAIN_DB:
                entries = ctrs[n_val_entries:]
            elif stage == constants.VAL_DB:
                entries = ctrs[:n_val_entries]
        elif stage == constants.TEST_DB:
            if self.userdata['validation_record'] != 'none':
                if self.userdata['test_image_file']:
                    raise ValueError("Specify either an image or a record from the validation set.")
                # test record from validation set
                entries = [ctrs[int(self.validation_record)]]

            else:
                # test image file
                entries = [self.userdata['test_image_file']]

        return entries

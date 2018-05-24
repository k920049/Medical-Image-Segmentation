from __future__ import absolute_import

import cv2
import numpy as np
import fnmatch
import os
import re
import pydicom

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
    extracted = []
    for elem in contours:
        extracted.append(Contour(elem))
    return extracted


def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    img = load_image(full_path)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    label = cv2.fillPoly(label, [ctrs], 1)
    img = img.astype(np.float32)
    label = label.astype(np.uint8)
    return img, label


def load_image(full_path):
    f = pydicom.dcmread(full_path)
    return f.pixel_array.astype(np.int)

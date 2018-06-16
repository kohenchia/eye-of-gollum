from __future__ import print_function

import os
import sys
import cv2
import random
import datetime
import time
import math
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from .net import Net
from .bbox import *

torch.backends.cudnn.bencmark = True


class Detector(object):

    def __init__(self):
        """
        Initializer
        """
        self.net = Net()
        self.net.load_state_dict(torch.load('s3fd/models/001.pth'))
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def detect(self, img):
        """
        Detects all the faces in the image and returns a list of bounding boxes

        :param img: numpy array of size (height, width, channels)
        :returns: A list of bounding boxes in the format (x1, y1, x2, y2)
        :rtype: list
        """
        img = img - np.array([104, 117, 123])       # Pixel normalization
        img = img.transpose(2, 0, 1)                # Convert to channel first
        img = img.reshape((1,) + img.shape)         # Batch of 1 image
        img = torch.from_numpy(img).float().cuda()  # Convert to tensor, put on GPU
        BB, CC, HH, WW = img.size()

        predictions = self.net(img)  # Money shot!

        bboxlist = []

        # Convert class predictions to softmax scores
        for i in range(int(len(predictions)/2)):
            predictions[i * 2] = F.softmax(predictions[i * 2], dim=1)

        # Convert location predictions to bounding boxes in the original image's scale
        for i in range(int(len(predictions)/2)):
            class_predictions = predictions[i * 2].data.cpu()
            bbox_predictions = predictions[i * 2 + 1].data.cpu()
            FB, FC, FH, FW = class_predictions.size()      # feature map size
            stride = 2 ** (i + 2)             # 4, 8, 16, 32, 64, 128

            # For each location in the feature map, decode it with the prior to get the predicted bounding box
            for hindex, windex in itertools.product(range(FH), range(FW)):
                axc = stride / 2 + windex * stride
                ayc = stride / 2 + hindex * stride
                score = class_predictions[0, 1, hindex, windex]
                loc = bbox_predictions[0, :, hindex, windex].contiguous().view(1, 4)

                if score < 0.05:
                    continue

                # (stride * 4) is the desired anchor size. Refer to arXiv paper.
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])

        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))

        # Run non-max supression against the bounding boxes
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]

        # Only use boxes with a score greater than 0.5
        bboxlist = bboxlist[bboxlist[:, -1] > 0.5, :]

        return bboxlist[:, :-1]
from __future__ import print_function

import logging
import numpy as np
import time

import torch
import torch.nn.functional as F

from .net import Net

torch.backends.cudnn.bencmark = True
LOG = logging.getLogger(__name__)


class Detector(object):

    def __init__(self, model_path=None):
        """
        Initializer
        """
        if not model_path:
            raise Exception('model_path must be a valid path.')

        self.net = Net()
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        if torch.cuda.is_available():
            self.net.cuda()

    def _predictions_to_bbox_list(self, predictions):
        """
        Converts predictions (S3FD output) to a bboxlist
        """
        _tic = time.time()
        bboxlist = []

        # -------------------------------------------------------------------------------------
        # Vectorized decoding of predicted locations to bboxes in the original image's scale
        #
        # predictions: [
        #   [B x 2 x H x W],      # class scores for feature map 1
        #   [B x 4 x H x W],      # location predictions for feature map 1
        #   [B x 2 x H x W],      # class scores for feature map 2
        #   [B x 4 x H x W],      # location predictions for feature map 2
        #   ...repeat...
        # ]
        # -------------------------------------------------------------------------------------
        num_feature_maps = int(len(predictions)/2)
        for i in range(num_feature_maps):

            # Get feature map size
            FB, FC, FH, FW = predictions[i * 2].size()

            # Get effective stride: 4, 8, 16, 32, 64, 128
            # (See arXiv paper for details)
            stride = 2 ** (i + 2)                       

            # Pick out the class scores from the second softmax channel into [H x W]
            softmax_scores = F.softmax(predictions[i * 2], dim=1)
            class_scores = softmax_scores[0, 1]
            assert class_scores.shape == (FH, FW)

            # Pick out all the location predictions into [4 x H x W]
            loc = predictions[i * 2 + 1][0]
            assert loc.shape == (4, FH, FW)

            # Generate feature map X, Y coordinates into two [H x W] arrays
            axc, ayc = np.meshgrid(                     
                np.arange(FW),
                np.arange(FH)
            )                          
            axc = torch.from_numpy(axc).cuda().type(loc.dtype)
            ayc = torch.from_numpy(ayc).cuda().type(loc.dtype)
            assert axc.shape == (FH, FW)
            assert ayc.shape == (FH, FW)

            # Pick out predictions with a minimum score, then:
            # Combine everything into a [N x 7] array where N is the number of predictions:
            # [
            #   [axc, ayc, loc_1, loc_2, loc_3, loc_4, score],
            #   [axc, ayc, loc_1, loc_2, loc_3, loc_4, score],
            #   [axc, ayc, loc_1, loc_2, loc_3, loc_4, score],
            #   ...repeat for N predictions...
            # ]
            mask = class_scores > 0.05
            num_predictions = torch.sum(mask)
            if num_predictions <= 0:
                continue

            bboxes = torch.stack(
                (
                    axc[mask],
                    ayc[mask],
                    loc[0][mask],
                    loc[1][mask],
                    loc[2][mask],
                    loc[3][mask],
                    class_scores[mask]
                ),
                dim=1
            )
            assert bboxes.shape == (num_predictions, 7)

            # Convert feature map coordinates to predicted bbox coordinates
            # (See arXiv paper for details)
            variances = [0.1, 0.2]
            bboxes[:, :2] *= stride                        # a_xy *= stride
            bboxes[:, :2] += (stride / 2)                  # a_xy += (stride / 2)
            bboxes[:, 2:4] *= (variances[0] * stride * 4)  # loc_xy * var * a_wh
            bboxes[:, :2] += bboxes[:, 2:4]                # a_xy += loc_xy
            bboxes[:, 4:6] *= variances[1]                 # loc_wh * var
            bboxes[:, 4:6].exp_()                          # loc_wh.exp()
            bboxes[:, 4:6] *= (stride * 4)                 # loc_wh * a_wh
            bboxes[:, :2] -= (bboxes[:, 4:6] / 2)          # x1y1 = a_xy - (loc_wh/2)
            bboxes[:, 4:6] += bboxes[:, :2]                # x2y2 = loc_wh + x1y1
            x1y1x2y2_list = bboxes[:, [0, 1, 4, 5, 6]]     # [x1, y1, x2, y2, score]
            assert x1y1x2y2_list.shape == (num_predictions, 5)

            # Add all the current feature map's predictions to the list
            bboxlist.append(x1y1x2y2_list)

        # Collect all bboxes
        bboxlist = torch.cat(bboxlist)
        if 0 == len(bboxlist):
            return []

        # Run non-max supression against the bounding boxes
        keep = self._nms(bboxlist.cpu().numpy(), 0.3)
        bboxlist = bboxlist[keep, :]

        # Only use boxes with a score greater than 0.5
        bboxlist = bboxlist[bboxlist[:, -1] > 0.5, :]
        if 0 == len(bboxlist):
            return []
        else:
            return bboxlist[:, :-1]

    def _nms(self, dets, thresh):
        """
        Non-max suppression
        TODO: Convert to Pytorch code
        """
        if 0 == len(dets):
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]),np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]),np.minimum(y2[i], y2[order[1:]])

            w, h = np.maximum(0.0, xx2 - xx1 + 1), np.maximum(0.0, yy2 - yy1 + 1)
            ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

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

        with torch.no_grad():
            _tic = time.time()
            predictions = self.net(img)
            _toc = time.time()
            LOG.debug('Forward pass took {}s'.format(_toc - _tic))

            _tic = time.time()
            bbox_list = self._predictions_to_bbox_list(predictions)
            _toc = time.time()
            LOG.debug('Processing bboxes took {}s'.format(_toc - _tic))

        return bbox_list


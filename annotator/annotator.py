import cv2
import numpy as np
from PIL import Image

from s3fd import detector

class Annotator(object):
    """
    Class that encapsulates a bounding box detector
    """

    def __init__(self, network='s3fd'):
        """
        Initializer
        """
        self.BOX_LINE_WIDTH = 3
        self.BOX_LINE_COLOR = (0, 255, 0)
        if network == 's3fd':
            self.detector = detector.Detector()
        else:
            raise Exception('Network {} not supported.'.format(network))

    def annotate(self, img):
        """
        Annotates the image with bounding boxes.

        :param img: A PIL image
        :returns: A PIL image with bounding boxes
        """
        arr = np.asarray(img)
        face_locations = self.detector.detect(arr)

        for (x1, y1, x2, y2) in face_locations:
            arr = cv2.rectangle(arr,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                self.BOX_LINE_COLOR,
                                self.BOX_LINE_WIDTH)

        return Image.fromarray(arr)
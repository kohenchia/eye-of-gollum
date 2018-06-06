"""Module that adds bounding boxes to video frames around recognized faces."""
import cv2
import face_recognition
import numpy as np
from PIL import Image

class Annotator(object):

    def __init__(self):
        """
        Initializer
        """
        self.BOX_LINE_WIDTH = 3
        self.BOX_LINE_COLOR = (0, 255, 0)

    def annotate(self, image):
        """
        Annotate frame with bounding boxes.
        """
        arr = np.asarray(image)

        # TODO: Replace with a working annotator
        # face_locations = face_recognition.face_locations(arr)
        FAKE_SIZE = 200
        FAKE_TOP = int(np.random.rand() * (1080 - FAKE_SIZE))
        FAKE_LEFT = int(np.random.rand() * (1920 - FAKE_SIZE))
        face_locations = [(FAKE_TOP, FAKE_LEFT, FAKE_TOP + FAKE_SIZE, FAKE_LEFT + FAKE_SIZE)]

        for (top, right, bottom, left) in face_locations:
            arr = cv2.rectangle(arr,
                                (left, top),
                                (right, bottom),
                                self.BOX_LINE_COLOR,
                                self.BOX_LINE_WIDTH)

        return Image.fromarray(arr)

"""IP Camera client that pulls frames from a video feed."""
import cv2
import logging

LOG = logging.getLogger(__name__)


class IPCam(object):

    def __init__(self, stream_url=None):
        """
        Initializer
        """
        if stream_url is None:
            raise Exception('Must provide a valid stream URL. Received: {}'.format(stream_url))

        self.capture = cv2.VideoCapture(stream_url)

    
    def get_next_frame(self, resize_dimensions=(640, 360)):
        """
        Gets the next frame from the IP camera as a numpy array.
        """
        ret, frame = self.capture.read()
        if not ret:
            logging.error('Unable to get frame from IP camera: {}'.format(ret))
            return False

        # Convert frame into RGB (from OpenCV's BGR format)
        rgb_frame = frame[:, :, ::-1]

        # Downsize frame for faster processing
        resized_rgb_frame = cv2.resize(rgb_frame, resize_dimensions)

        return resized_rgb_frame

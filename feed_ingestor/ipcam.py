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
    
    def get_next_frame(self):
        """
        Gets the next frame from the IP camera as a numpy array.
        """
        ret, frame = self.capture.read()
        if not ret:
            return None

        # Convert frame into RGB (from OpenCV's BGR format)
        rgb_frame = frame[:, :, ::-1]
        return rgb_frame
    
    def __enter__(self):
        """
        Entry function for context statements
        """
        return self

    def __exit__(self):
        """
        Shuts down the video capture gracefully
        """
        self.capture.release()

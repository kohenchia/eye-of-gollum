import numpy as np
import os
import pytest
import sys
from PIL import Image

tests_dir = os.path.dirname(__file__)
model_dir = os.path.dirname(os.path.dirname(__file__)) + '/models'


@pytest.fixture(scope='module')
def detector():
    """
    Create the detector instance as a pytest fixture
    """
    from ..detector import Detector
    return Detector(f'{model_dir}/001.pth')


def test_import(detector):
    """
    Make sure we can import modules and load the detector model
    """
    assert detector is not None


def test_detector(detector):
    """
    Make sure we can run a forward pass on an image
    """
    i = Image.open(f'{tests_dir}/f.jpg')
    bbox_list = detector.detect(np.asarray(i))
    assert len(bbox_list) >= 1
    assert len(bbox_list[0]) == 4
    assert bbox_list[0][1] > 0
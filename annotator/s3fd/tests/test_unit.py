import numpy as np
import os
import pytest
import sys
import torch
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
    Make sure we can run a forward pass on an image, and the predictions remain the same.
    """
    i = Image.open(f'{tests_dir}/f.jpg')
    predicted_bbox_list = detector.detect(np.asarray(i))
    # Expected bbox values were retrived from a forward pass
    # using the original (inefficient) double-loop code.
    expected_bbox_list = torch.tensor(
        [[48.525726, 286.9632, 86.64511, 335.23447],
         [266.67667, 200.41855, 298.0504, 244.6362],
         [490.83972, 152.88602, 516.1749, 189.27315],
         [330.67038, 247.75795, 366.0774, 298.80145],
         [337.8342, 167.22401, 359.92166, 196.30225],
         [553.79156, 51.886246, 568.62244, 74.27113],
         [366.9683, 134.5618, 390.72498, 161.17648]]
    ).cuda().type(predicted_bbox_list.dtype)
    assert len(predicted_bbox_list) >= 1
    assert len(predicted_bbox_list[0]) == 4
    assert torch.equal(expected_bbox_list, predicted_bbox_list)

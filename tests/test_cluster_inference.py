# -*- coding: utf-8 -*-
"""
Testing cluster inference functions
"""
import numpy as np
import pyrft as pr

def test_find_clusters():
    """ Testing the find_clusters function """
    data = np.array([[ 1, 0, 1],[ 1, 1, 0]])
    threshold = 0.5
    for below in np.arange(1):
        cluster_image, _ = pr.find_clusters(data, threshold, below = below)
        assert cluster_image.shape == data.shape

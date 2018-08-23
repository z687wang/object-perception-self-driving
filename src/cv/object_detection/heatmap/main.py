import numpy as np
import cv2
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt

def add_heat(img, bbox_list, show_threshold=False):
    heatmap = np.zeros_like(img[:,:,0])
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    if (show_threshold):
        cv2.imshow('heatmap', heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    heat_map = apply_threshold(heatmap, 1)
    if (show_threshold):
        cv2.imshow('heatmap_threshold', heat_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    label_heat_map = label(heat_map)
    return label_heat_map


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


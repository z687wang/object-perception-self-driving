import numpy as np
import cv2
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt

def apply_heat_map(img, bbox_list, show_threshold=False):
    heatmap = add_heat(img, bbox_list)
    # plt.imshow(heatmap, cmap='hot')
    # plt.show()
    if (show_threshold):
        cv2.imshow('heatmap', heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    heat_map = apply_threshold(heatmap, 5)
    filtered_bbox_list = filter_bbox(heat_map, bbox_list, 20)
    filter_heatmap = add_heat(img, filtered_bbox_list)

    if (show_threshold):
        cv2.imshow('heatmap_threshold', filter_heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    label_heat_map = label(filter_heatmap)
    return label_heat_map

def add_heat(img, bbox_list):
    heatmap = np.zeros_like(img[:,:,0])
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def filter_bbox(heat_map, bbox_list, threshold=2):
    filtered_bbox = []
    for bbox in bbox_list:
        filter_bbox = 0
        for row in heat_map[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]:
            for ptn in row:
                if (ptn >= 1):
                    filter_bbox += 1
        if (filter_bbox >= threshold):
            filtered_bbox.append(bbox)
    return filtered_bbox


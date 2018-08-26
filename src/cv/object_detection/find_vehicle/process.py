import cv2
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

import sys
sys.path.append('../')

from object_detection.find_vehicle.main import find_vehicles, slide_window, search_windows
from object_detection.train import train, load_train_data
from object_detection.heatmap.main import apply_heat_map, apply_threshold
from object_detection.draw.main import draw_boxes, draw_labeled_bboxes

draw_val = 0

def detect_vehicles(img, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, bin_spatial_feat, color_hist_feat, hog_feat, load_model):
  svc, X_scaler = train(colorspace, orient, pix_per_cell, cell_per_block, hog_channel, load_model, bin_spatial_feat, color_hist_feat, hog_feat)
  rectangles = []

  scanning_config = [[400, 464, 1.0], [416, 480, 1.0], [400, 496, 1.5], [432, 528, 1.5], [400, 528, 1.5], [400, 528, 2.0], [432, 560, 2.0], [400, 596, 3.5], [464, 660, 3.5]]


  for config in scanning_config:
    ystart = config[0]
    ystop = config[1]
    scale = config[2]
    rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient,
                                    pix_per_cell, cell_per_block, spatial_size, hist_bins))
  rectangles = [item for sublist in rectangles for item in sublist]

  # draw_img = draw_boxes(np.copy(img), rectangles, 'random', 1)
  # if (draw_val >= 30):
  #   cv2.imshow('rectangles', draw_img)
  #   cv2.waitKey(0)
  #   cv2.destroyAllWindows()
  #   draw_val = 0
  # else:
  #   draw_val = draw_val + 1


  labels = apply_heat_map(img, rectangles, False)
  
  img_rects, rects = draw_labeled_bboxes(img, labels, random_color=True)
  
  return img_rects, rectangles
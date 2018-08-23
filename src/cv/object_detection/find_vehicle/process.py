import cv2
import numpy as np
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

import sys
sys.path.append('../')

from find_vehicle.main import find_vehicles, slide_window, search_windows
from train import train, load_train_data
from draw.main import draw_boxes
from heatmap.main import add_heat, apply_threshold
from draw.main import draw_boxes, draw_labeled_bboxes

def detect_vehicles(img, colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, bin_spatial_feat, color_hist_feat, hog_feat, load_model):
  svc, X_scaler = train(colorspace, orient, pix_per_cell, cell_per_block, hog_channel, load_model, bin_spatial_feat, color_hist_feat, hog_feat)
  rectangles = []

  ystart = 400
  ystop = 464
  scale = 1.0
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  ystart = 416
  ystop = 480
  scale = 1.0
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  ystart = 400
  ystop = 496
  scale = 1.5
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))

  ystart = 432
  ystop = 528
  scale = 1.0
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  ystart = 432
  ystop = 528
  scale = 1.5
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  ystart = 400
  ystop = 528
  scale = 2.0
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  ystart = 432
  ystop = 560
  scale = 2.0
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  ystart = 400
  ystop = 596
  scale = 3.5
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  ystart = 464
  ystop = 660
  scale = 3.5
  rectangles.append(find_vehicles(img, colorspace, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial_size, hist_bins))
  rectangles = [item for sublist in rectangles for item in sublist] 
  
  labels = add_heat(img, rectangles, False)
  
  img_rects, rects = draw_labeled_bboxes(img, labels, random_color=True)
  
  return img_rects
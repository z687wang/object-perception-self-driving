import cv2
import numpy as np
import sys
sys.path.append('../')
from object_detection.find_vehicle.process import detect_vehicles
from moviepy.editor import VideoFileClip

colorspace = 'YCrCb'
orient = 15
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (32, 32)
hist_bins = 32
bin_spatial_feat = True
color_hist_feat = True
hog_feat = True
load_model = True


def detect_objects(image):
  test_img_rects, rectangles = detect_vehicles(np.copy(image), colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, bin_spatial_feat, color_hist_feat, hog_feat, load_model)
  return test_img_rects

# test_img_rects, _ = detect_vehicles(np.copy(test_img), colorspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, bin_spatial_feat, color_hist_feat, hog_feat, load_model)
#
#
# cv2.imshow('draw', test_img_rects)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# test_out_file = '../../../combined_video_out.mp4'
# clip_test = VideoFileClip('../../../combined_video.mp4')
# clip_test_out = clip_test.fl_image(detect_objects)
# clip_test_out.write_videofile(test_out_file, audio=False)
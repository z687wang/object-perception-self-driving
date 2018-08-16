import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class TemplateMach:
  def __init__(self, color=(0, 0, 255), thick=6):
    self.color = color
    self.thick = thick
  
  def draw_boxes(self, img, bboxes, color=self.color, thick=self.thick):
    img_copy = np.copy(img)
    for bbox in bboxes:
      cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy
  
  def find_matches(self, img, template_list):
    bbox_list = []
    cv2.matchTemplate()
    cv2.minMaxLoc()


    return bbox_list


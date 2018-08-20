import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class TemplateMach:
  def __init__(self, color=(0, 0, 255), thick=6):
    self.color = color
    self.thick = thick
    self.method = cv2.TM_CCOEFF_NORMED
  
  def draw_boxes(self, img, bboxes):
    img_copy = np.copy(img)
    for bbox in bboxes:
      cv2.rectangle(img_copy, bbox[0], bbox[1], self.color, self.thick)
    return img_copy
  
  def find_matches(self, img, template_list):
    bbox_list = []

    for templateSrc in template_list:
      template = mpimg.imread(templateSrc)
      res = cv2.matchTemplate(img, template, self.method)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      if self.method in [cv2.TM_SQDIFF, cv2.TM_CCOEFF_NORMED]:
        top_left = min_loc
      else:
        top_rigth = max_loc

      w = img.shape[0]
      h = img.shape[1]
      bottom_right = (top_left[0] + w, top_left[1] + h)
      bbox_list.append((top_left, bottom_right))
    return bbox_list

if __name__ == '__main__':
  img = cv2.imread('../../../../test_images/test1.jpg')
  templist = ['../../../../test_images/test2.jpg']
  tm = TemplateMach()
  bboxes = tm.find_matches(img, templist)
  result = tm.draw_boxes(img, bboxes)
  cv2.imshow('image', result)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
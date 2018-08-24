import numpy as np
import cv2

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            print(color)
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), thick)
    # Return the image copy with boxes drawn
    return imcopy

def draw_labeled_bboxes(img, labels, random_color=True):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        if random_color:
            color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        else:
            color = (0,0,255)
        print('color:', color)
        cv2.rectangle(img, (bbox[0][0], bbox[0][1] + 10), bbox[1], (0,255,0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'Vehicle Detected', bbox[0], font, 0.5, (0,255,0), 1, cv2.LINE_AA)
    # Return the image and final rectangles
    return img, rects
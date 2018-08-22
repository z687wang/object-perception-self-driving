from skimage.feature import hog
import cv2

class HOG:
    def __init__(self, orient=9, pix_per_cell=8, cell_per_block=2):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block

    def get_hog_features(self, img, vis=False, feature_vec=False):

        if vis == True:
            features, hog_image = hog(img, orientations=self.orient, 
                                    pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                    cells_per_block=(self.cell_per_block, self.cell_per_block), 
                                    transform_sqrt=False,
                                    block_norm='L2-Hys',
                                    visualize=vis, feature_vector=feature_vec)
            return features, hog_image

        else:      
            features = hog(img, orientations=self.orient, 
                        pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                        cells_per_block=(self.cell_per_block, self.cell_per_block), 
                        transform_sqrt=False,
                        block_norm='L2-Hys',
                        visualize=vis, feature_vector=feature_vec)
            return features


if __name__ == '__main__':
    imgSrc = '../../../../test_images/test1.jpg'
    img = cv2.imread(imgSrc)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Hog = HOG(9, 8, 2)
    hog_features, hog_img = Hog.get_hog_features(gray)
    cv2.imshow('HOG', hog_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

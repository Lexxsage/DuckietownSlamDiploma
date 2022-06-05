#The images from the simulator will always be distorted,
#since the images from the real duckiebot will also be distorted.
#So, we need to rectify it

import yaml
import numpy as np
import os
import cv2 as cv2

class ImageStraightener():

    def __init__(self):
        #from where we get images
        self.images_dir = "Images/"
        #where we save images
        self.undistorted_images_dir = "Undistored/"

    def get_all_images(self):
        images_filenames = sorted(os.listdir(self.images_dir))
        return images_filenames

    def get_parameters(self):
        with open("duck666_instricts.yaml") as file:
            camera_list = yaml.load(file,Loader = yaml.FullLoader)
        camera_intrinsic_matrix = np.array(camera_list['camera_matrix']['data']).reshape(3,3)
        distortion_coefficient = np.array(camera_list['distortion_coefficients']['data']).reshape(5,1)
        return camera_intrinsic_matrix, distortion_coefficient

    def undistort_images(self):
        images_filenames = self.get_all_images()
        camera_intrinsic_matrix, distortion_coefficient = self.get_parameters()

        image_1 = cv2.imread(self.images_dir + images_filenames[0])
        height = image_1.shape[0]
        width = image_1.shape[1]
        newmatrix, roi = cv2.getOptimalNewCameraMatrix(camera_intrinsic_matrix,distortion_coefficient,(width,height),1, (width,height))
        map_x, map_y = c

        for i in images_filenames:
            print(images_data_dir + i)
            img = cv2.imread(images_dir +i)
            new_image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imwrite(undistorted_images_dir + "undistorted_%s" %i, new_image)
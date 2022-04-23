#The images from the simulator will always be distorted,
#since the images from the real duckiebot will also be distorted.
#So, we need to rectify it

import yaml
import numpy as np
import os
import cv2 as cv2
with open("intrinsic.yaml") as file: # what the yaml?
    camera_list = yaml.load(file,Loader = yaml.FullLoader)

camera_intrinsic_matrix = np.array(camera_list['camera_matrix']['data']).reshape(3,3)

distortion_coefficient = np.array(camera_list['distortion_coefficients']['data']).reshape(5,1)

#from where we get images
images_dir = " "
#where we save images
undistorted_images_dir = " "

def save_images_dir(images_dir, dir_name):
    images_dir = dir_name

def save_undistorted_imaged_dir(undistorted_images_dir, dir_name):
    undistorted_images_dir = dir_name

def get_all_images(images_data_dir):
    images_filenames = sorted(os.listdir(images_data_dir))
    return images_filenames

def undistort_images(images_data_dir, images_filenames, camera_intrinsic_matrix, distortion_coefficient):
    image_1 = cv2.imread(images_data_dir + images_filenames[0])
    height = image_1.shape[0]
    width = image_1.shape[1]
    newmatrix, roi = cv2.getOptimalNewCameraMatrix(camera_intrinsic_matrix,distortion_coefficient,(width,height),1, (width,height))
    map_x, map_y = c

    for i in images_filenames:
        print(images_data_dir + i)
        img = cv2.imread(images_dir +i)
        new_image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        cv2.imwrite(undistorted_images_dir + "undistorted_%s" %i, new_image)
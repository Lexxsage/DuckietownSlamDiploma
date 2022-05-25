#!/usr/bin/env python

import os
import rospy
import rosbag
import cv2
#from duckietown_utils import rgb_from_ros

class BagToTxtConverter():
    def __init__(self):
        self.bagFile = rosbag.Bag("TestBag.bag")
        self.txtFile = "TestBag.txt"
        self.imageCount = 0

    def convertOdometryBagToTxt(self):
        f = open(self.txtFile, "a")
        for topic, msg, t  in self.bagFile.read_messages(topics = ["/duck666/wheels_driver_node/wheels_cmd"]):
            f.write(str(msg.vel_left) +" " + str(msg.vel_right) + " " + str(msg.header.stamp) + " \n")
        self.bagFile.close()


    def convertImagesBagToJpg(self):
        for topic,msg, t in self.bagFile.read_messages(topics = ["/duck666/camera_node/camera_info"]):
            cv_img = rgb_from_ros(msg)
            cv2.imwrite(os.path.join("/Images", "frame%06i.png" %count), cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
            self.imageCount +=1
        print(self.imageCount)
        self.bagFile.close()


    def writeImageTimeStamp(self):
        f = open(self.txtFile, "a")
        for topic,msg, t in self.bagFile.read_messages(topics = ["/duck666/camera_node/image/compressed"]):
            f.write(str(msg.header.stamp) +" \n")
        self.bagFile.close()

converter = BagToTxtConverter()
converter.convertOdometryBagToTxt()
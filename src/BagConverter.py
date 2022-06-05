#!/usr/bin/env python

import os
import rospy
import rosbag
import cv2
#pip install duckietown-utils-daffy
import duckietown_code_utils as dtu
import duckietown_rosdata_utils as dru

class BagToTxtConverter():
    def __init__(self):
        self.bagFile = rosbag.Bag("example.bag")
        self.wheelTxtFile = "TestBag.txt"
        self.stampTxtFile = "Stamp.txt"
        self.imageCount = 0

    def convertOdometryBagToTxt(self):
        f = open(self.wheelTxtFile, "a")
        for topic, msg, t  in self.bagFile.read_messages(topics = ["/duck666/wheels_driver_node/wheels_cmd"]):
            f.write(str(msg.vel_left) +" " + str(msg.vel_right) + " " + str(msg.header.stamp) + " \n")
        self.bagFile.close()


    def convertImagesBagToJpg(self):
        for topic,msg, t in self.bagFile.read_messages(topics = ["/duck666/camera_node/image/compressed"]):
            cv_img = dru.rgb_from_ros(msg)
            cv2.imwrite(os.path.join("/Images", "frame%06i.png" %count), cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
            self.imageCount +=1
        print(self.imageCount)
        self.bagFile.close()


    def writeImageTimeStamp(self):
        f = open(self.stampTxtFile, "a")
        for topic,msg, t in self.bagFile.read_messages(topics = ["/duck666/camera_node/image/compressed"]):
            f.write(str(msg.header.stamp) +" \n")
        self.bagFile.close()

    def convertBagFile(self):
        self.convertOdometryBagToTxt()
        self.convertImagesBagToJpg()
        self.writeImageTimeStamp()

converter = BagToTxtConverter()
converter.convertOdometryBagToTxt()
#!/usr/bin/env python

import os
import rospy
import rosbag
import cv2
from duckietown_utils import rgb_from_ros

bagFile = rosbag.Bag("Test.bag")
imageCount = 0

def chooseBagFile(bagName):
    bagFile = rosbag.Bag(bagName)


def convertOdometryBagToTxt():
    print(bagFile)
    print("msg.header")
    for topic, msg, t  in bagFile.read_messages(topics = ["/duck666/wheels_driver_node/wheels_cmd"]):
        print(msg.header)
        f.write(str(msg.vel_left) +" " + str(msg.vel_right) + " " + str(msg.header.stamp) + " \n")
    bagFile.close()


def convertImagesBagToJpg():
    for topic,msg, t in bagFile.read_messages(topics = ["/duck666/camera_node/image/compressed"]):
        print(msg.header)
        cv_img = rgb_from_ros(msg)
        cv2.imwrite(os.path.join("/Images", "frame%06i.png" %count), cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
        imageCount +=1
    print(imageCount)
    bagFile.close()



def writeImageTimeStamp():
    for topic,msg, t in bag.read_messages(topics = ["/duck666/camera_node/image/compressed"]):
        print(msg.header)
        print(msg.header.stamp)
        f.write(str(msg.header.stamp) +" \n")
    bagFile.close()



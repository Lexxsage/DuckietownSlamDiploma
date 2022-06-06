#!/usr/bin/env python
import rospy
import roslib
from image_straightening import ImageStraightener
from odometry_mapper import OdometryMapper
from std_msgs.msg import ByteMultiArray

class ImageMapperListener():

    def __init__(self):
        # Initialize the node with rospy
        rospy.init_node('image_mapper_subscriber_node')
        self.imageStraightener = ImageStraightener()

    def callback(self, data):
        self.imageStraightener.get_parameters()

    def spin_mapper(self):
        # Create subscriber
        self.subscriber = rospy.Subscriber("~imageAdapterTopic", ByteMultiArray, self.callback)
        # spin to keep the script for exiting
        rospy.spin()

class OdometryMapperListener():

    def __init__(self):
        # Initialize the node with rospy
        rospy.init_node('odometry_mapper_subscriber_node')
        self.odometryMapper = OdometryMapper()

    def callback(event):
        self.odometryMapper.create_map_odometry()

    def spin_mapper():
        # Create subscriber
        self.subscriber = rospy.Subscriber("~odometryAdapterTopic", ByteMultiArray, self.callback)
        # spin to keep the script for exiting
        rospy.spin()


def create_mapper_subscriber(isOdometry):
    if (isOdometry):
        odometryMapper = OdometryMapperListener()
        odometryMapper.spin_mapper()
    else:
        imageMapper = ImageMapperListener()
        imageMapper.spin_mapper()


if __name__ == '__main__':
    try:
        create_mapper_subscriber(true)
    except rospy.ROSInterruptException:
        pass
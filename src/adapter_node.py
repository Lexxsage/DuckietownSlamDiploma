#!/usr/bin/env python
import rospy
import roslib
from images_adapter import ImageAdapter
from odometry_adapter import OdometryAdapter
from std_msgs.msg import ByteMultiArray

def ImageAdapterNode():
    def init():
        # Initialize the node with rospy
        rospy.init_node('adapter_node')
        # Create publisher
        publisher = rospy.Publisher("~imageTopic", ByteMultiArray,queue_size=100)

    def get_image_dir():
        DIR = roslib.packages.get_pkg_dir("duckietown_slam", required=True)

    def callback(event):
        global publisher
        msg = ByteMultiArray()
        input_folder = str(input())
        folder = DIR + input_folder
        msg.data = ImageAdapter.load_all_images_from_folder(folder)
        publisher.publish(msg)

    def spin_adapter():
        # Read parameter
        pub_period = rospy.get_param("~pub_period",1.0)
        # Create timer
        rospy.Timer(rospy.Duration.from_sec(pub_period),callback)
        # spin to keep the script for exiting
        rospy.spin()

def OdomtryAdapterNode():
    def init():
        # Initialize the node with rospy
        rospy.init_node('adapter_node')
        # Create publisher
        publisher = rospy.Publisher("~odometryTopic", ByteMultiArray,queue_size=100)

    def callback(event):
        global publisher
        msg = ByteMultiArray()
        folder = str(init())
        msg.data
        w, y, theta = OdometryAdapter.load_odometry(folder)
        publisher.publish(msg)

    def spin_adapter():
        # Read parameter
        pub_period = rospy.get_param("~pub_period",1.0)
        # Create timer
        rospy.Timer(rospy.Duration.from_sec(pub_period),callback)
        # spin to keep the script for exiting
        rospy.spin()


def create_adapter_node(isOdometry):
    if (isOdometry == 'y'):
        OdometryAdapter.init()
        OdometryAdapter.spin_adapter()
    else:
        ImageAdapterNode.init()
        ImageAdapterNode.get_image_dir()
        ImageAdapterNode.spin_adapter()


if __name__ == '__main__':
    try:0
        isOdometry = str(input())
        creta()
    except rospy.ROSInterruptException:
        pass
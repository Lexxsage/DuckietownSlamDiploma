#!/usr/bin/env python
import rospy
import roslib
from images_adapter import ImageAdapter
from odometry_adapter import OdometryAdapter
from std_msgs.msg import ByteMultiArray

class ImageAdapterNode():

    def __init__(self):
        # Initialize the node with rospy
        rospy.init_node('adapter_node')
        # Create publisher
        self.publisher = rospy.Publisher("~imageTopic", ByteMultiArray,queue_size=100)
        self.dir = roslib.packages.get_pkg_dir("duckietown_slam", required=True)
        self.input_folder = "dataset_dt"


    def callback(self, event):
        msg = ByteMultiArray()
        folder = self.dir + self.input_folder
        msg.data = ImageAdapter.load_all_images_from_folder(folder)
        self.publisher.publish(msg)

    def spin_adapter(self):
        # Read parameter
        pub_period = rospy.get_param("~pub_period",1.0)
        # Create timer
        rospy.Timer(rospy.Duration.from_sec(pub_period),callback)
        # spin to keep the script for exiting
        rospy.spin()

class OdometryAdapterNode():

    def __init__(self):
        # Initialize the node with rospy
        rospy.init_node('adapter_node')
        # Create publisher
        self.publisher = rospy.Publisher("~odometryTopic", ByteMultiArray,queue_size=100)
        self.folder = "Text.txt"

    def callback(event):
        msg = ByteMultiArray()
        msg.data
        w, y, theta = OdometryAdapter.load_odometry(self.folder)
        publisher.publish(msg)

    def spin_adapter():
        # Read parameter
        pub_period = rospy.get_param("~pub_period",1.0)
        # Create timer
        rospy.Timer(rospy.Duration.from_sec(pub_period),callback)
        # spin to keep the script for exiting
        rospy.spin()


def create_adapter_node(isOdometry):
    if (isOdometry):
        OdometryAdapter.init()
        OdometryAdapter.spin_adapter()
    else:
        ImageAdapterNode.init()
        ImageAdapterNode.spin_adapter()


if __name__ == '__main__':
    try:
        create_adapter_node(true)
    except rospy.ROSInterruptException:
        pass
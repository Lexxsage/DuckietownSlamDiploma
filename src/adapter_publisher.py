#!/usr/bin/env python
import rospy
import roslib
from images_adapter import ImageAdapter
from odometry_adapter import OdometryAdapter
from std_msgs.msg import ByteMultiArray

class ImageAdapterPublisher():

    def __init__(self):
        # Initialize the node with rospy
        rospy.init_node('image_adapter_publisher_node')
        # Create publisher
        self.publisher = rospy.Publisher("~imageAdapterTopic", ByteMultiArray,queue_size=100)
        self.dir = roslib.packages.get_pkg_dir("duckietown_slam", required=True)
        self.input_folder = "dataset_dt"
        self.imageAdapter = ImageAdapter()


    def callback(self, event):
        msg = ByteMultiArray()
        folder = self.dir + self.input_folder
        msg.data = self.imageAdapter.load_all_images_from_folder(folder)
        self.publisher.publish(msg)

    def spin_adapter(self):
        # Read parameter
        pub_period = rospy.get_param("~pub_period",1.0)
        # Create timer
        rospy.Timer(rospy.Duration.from_sec(pub_period), self.callback)
        # spin to keep the script for exiting
        rospy.spin()

class OdometryAdapterPublisher():

    def __init__(self):
        # Initialize the node with rospy
        rospy.init_node('odometry_adapter_publisher_node')
        # Create publisher
        self.publisher = rospy.Publisher("~odometryAdapterTopic", ByteMultiArray,queue_size=100)
        self.folder = "Text.txt"
        self.odometryAdapter = OdometryAdapter()

    def callback(event):
        msg = ByteMultiArray()
        msg.data
        w, y, theta = self.odometryAdapter.load_odometry(self.folder)
        publisher.publish(msg)

    def spin_adapter():
        # Read parameter
        pub_period = rospy.get_param("~pub_period",1.0)
        # Create timer
        rospy.Timer(rospy.Duration.from_sec(pub_period), self.callback)
        # spin to keep the script for exiting
        rospy.spin()


def create_adapter_publisher(isOdometry):
    if (isOdometry):
        odometryAdapter = OdometryAdapterPublisher()
        odometryAdapter.spin_adapter()
    else:
        imageAdapter = ImageAdapterPublisher()
        imageAdapter.spin_adapter()


if __name__ == '__main__':
    try:
        create_adapter_node(true)
    except rospy.ROSInterruptException:
        pass
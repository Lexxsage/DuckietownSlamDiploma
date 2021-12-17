#!/usr/bin/env python
import rospy
import roslib
import adapter
from std_msgs.msg import ByteMultiArray
# Initialize the node with rospy
rospy.init_node('adapter_node')
# Create publisher
publisher = rospy.Publisher("~topic", ByteMultiArray,queue_size=100)

DIR = roslib.packages.get_pkg_dir("duckietown_slam", required=True)

# Define Timer callback
def callback(event):
	msg = ByteMultiArray()
	folder = DIR + "/dataset_dt/images/"
	msg.data = adapter.load_all_images_from_folder(folder)
	publisher.publish(msg)
# Read parameter
pub_period = rospy.get_param("~pub_period",1.0)
# Create timer
rospy.Timer(rospy.Duration.from_sec(pub_period),callback)
# spin to keep the script for exiting
rospy.spin()

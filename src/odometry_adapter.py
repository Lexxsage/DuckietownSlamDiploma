import matplotlib.pyplot as plt
import numpy as np

import sys

class OdometryAdapter():
    def __init__(self):
        self.fileName = "TestBag.txt"

    def load_odometry(self):
        sys.setrecursionlimit(100000000)
        #txt file of odometry
        with open(self.fileName) as f:
            content = f.readlines()
        vel_left =[]
        vel_right =[]
        time = []
        for line in content:
            line_split = line.split(" ")
            left = float(line_split[0])
            vel_left.append(left)
            right = float(line_split[1])
            vel_right.append(right)
            time.append(float(line_split[2])/10**9)

        print()
        time_diff = np.diff(time)
        return vel_left, vel_right, time_diff
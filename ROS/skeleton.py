#!/usr/bin/env python

"""Mandatory imports"""
import rospy
from sensor_msgs.msg import Image


"""Packets imports (Example)"""

import cv2
import numpy as np

from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg

#When you create the workspace you need to set what type of messages you need for the project.

"""Basic structure"""

#don't change the name of the function (callback)
def callback(data):
    #This could be change if you need it.
    im = np.frombuffer(data.data, dtype = np.uint8).reshape(data.height, data.width, -1)
    img = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

    """For publish"""
    #an array
    posEnv = Float32MultiArray(data = array)
    pub.publish(posEnv)
    
    #the image
    pub.publish(img)

    rate.sleep()


#You can change the name "cosa"
def COSA():
    #it's a good practica to use the same name.
    rospy.init_node('cosa') #or rospy.init_node('cosa', anonymous = True) # anonymoys add random numbers at the end of the name

    #Initialize publisher
    pub = rospy.Publisher('cosa', Float32MultiArray, queue_size = 1) # rospy.Publisher(name,Type of message,size of queue[mandatory to set it])

    #Initialize subscriber
    sub = rospy.Subscriber('cosa', numpy_msg(Image), callback) #rospy.Subscriber(name of the topic, Type of message, the function that receive the data)
    """
    Tip: write in other console 'rostopic list' --> that shows you the current topics
         write 'rostopic topicName info' --> that shows the type of the message
    """
    rospy.spin()

if __name__ == '__main__':
    COSA()

    

#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('terrapn')
import sys
import time
import math
import numpy as np
import rospy
import cv2
import tf
import geometry_msgs
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry, Path
from cv_bridge import CvBridge, CvBridgeError

class Dataset_Subscriber:
	def __init__(self):
		self.bridge = CvBridge()
		self.img_topic_name = "/camera/color/image_raw"
		self.imu_topic_name = "/imu/data_raw"
		self.odom_topic_name = "/odometry/filtered"
		self.gt_topic_name = "/laser_odom_to_init"
		
		# self.surface_name = "red_tiles"
		# self.surface_name = "smooth_tiles"
		# self.surface_name = "grass"
		# self.surface_name = "leaves"
		self.surface_name = "asphalt"

		# Topic names
		self.image_sub = rospy.Subscriber(self.img_topic_name, Image, self.img_callback)
		self.gt_sub = rospy.Subscriber(self.gt_topic_name, Odometry, self.gt_callback)
		# self.imu_sub = rospy.Subscriber(self.imu_topic_name, Imu, self.imu_callback)
		# self.odom_sub = rospy.Subscriber(self.odom_topic_name', Odometry, self.odometry_callback)

		# List lengths related to inputs and labels 
		self.patch_side = 200
		self.vel_vector_len = 150
		self.diff_duration = 12
		self.iter = 0

		# Ground Truth Attributes
		self.gt_XY = []
		self.gt_yaw = []

		# Wheel Odometry Attributes
		self.odom_XY = []
		self.odom_yaw = []

		# Instantaneous distance and yaw differences between odometry and ground truth
		self.dist_diff_instant = 0.0
		self.yaw_diff_instant = 0.0

		# Label list
		self.label = [0.0, 0.0, 0.0, 0.0]

		# Input velocity vector
		self.vels = []

		# IMU vector used for PCA variance calculation
		self.imu_linear = []
		self.imu_angular = []
		self.orientation = []

		# Lists to save distance and yaw differences and correspodning velocities (for viz)
		self.dist_diff = []
		self.yaw_diff = []
		self.vels2 = []
		
		

		self.t1 = time.time()
		print("Constructor Done for v2!")


	def img_callback(self,img_data):
		try:
			
			cv_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")

		except CvBridgeError as e:
			print(e)

		# Obtain patch
		(rows,cols,channels) = cv_image.shape
		cropped_image = cv_image[rows-self.patch_side:rows, cols/2 - self.patch_side/2:cols/2 + self.patch_side/2]

		# Obtain Odom and IMU data for this instant and append on to a vector
		odom_data = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=None)
		imu_data = rospy.wait_for_message(self.imu_topic_name, Imu, timeout=None)

		self.vels.append([odom_data.twist.twist.linear.x, odom_data.twist.twist.angular.z])
		self.imu_linear.append([imu_data.linear_acceleration.x,imu_data.linear_acceleration.y,imu_data.linear_acceleration.z])
		self.imu_angular.append([imu_data.angular_velocity.x,imu_data.angular_velocity.y,imu_data.angular_velocity.z])
		self.orientation.append([imu_data.orientation.x,imu_data.orientation.y,imu_data.orientation.z,imu_data.orientation.w])
		
		# print("Length of Velocity and IMU vectors = ", len(self.imu_linear)) #len(self.vels)
		if(len(self.odom_yaw) == self.diff_duration):
			print("Length of Odometry error vector = ", self.diff_duration)

			if len(self.vels) > self.vel_vector_len:
				# Delete the oldest velocity and IMU data
				self.vels.pop(0)
				self.imu_linear.pop(0)
				self.imu_angular.pop(0)
				print("Length of input velocities and IMU = ", len(self.vels))
				self.iter = self.iter + 1
				
				X = np.concatenate((self.imu_linear,self.imu_angular),axis=1)
				
				# Apply PCA on the data
				X_reduced = self.PCA(X,2)
				variance = np.var(X_reduced,axis=0)
				self.label[0] = variance[0]
				self.label[1] = variance[1]
				self.label[2] = self.dist_diff_instant
				self.label[3] = self.yaw_diff_instant
				# print("Label Vector = ", self.label)

				# Save dataset
				cv2.imwrite("/home/adarshjs/catkin_ws/src/terrapn/dataset/" + self.surface_name + "/" + str(self.iter) + ".png", cv_image)
				np.save("/home/adarshjs/catkin_ws/src/terrapn/dataset/" + self.surface_name + "/" + "input_velocity_" + str(self.iter) + ".npy", self.vels)
				np.save("/home/adarshjs/catkin_ws/src/terrapn/dataset/" + self.surface_name + "/" +"label_" + str(self.iter) + ".npy", self.label)




		
		# t2 = time.time()
		# print("Time to execute one loop of Image callback = ", t2 - self.t1)
		cv2.imshow("Image window", cropped_image)
		cv2.waitKey(1)


	def gt_callback(self, gt_data):
		
		# Record Ground truth data
		self.gt_XY.append([gt_data.pose.pose.position.x, gt_data.pose.pose.position.y])
		gt_quaternion = (gt_data.pose.pose.orientation.x, gt_data.pose.pose.orientation.y, gt_data.pose.pose.orientation.z, gt_data.pose.pose.orientation.w)
		euler = tf.transformations.euler_from_quaternion(gt_quaternion)
		self.gt_yaw.append(euler[2])

		# Record Odometry data
		odom_data = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=None)
		self.odom_XY.append([odom_data.pose.pose.position.x, odom_data.pose.pose.position.y])
		odom_quaternion = (odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y, odom_data.pose.pose.orientation.z, odom_data.pose.pose.orientation.w)
		euler = tf.transformations.euler_from_quaternion(odom_quaternion)
		self.odom_yaw.append(euler[2])

		if(len(self.odom_yaw) > self.diff_duration):
			del self.gt_XY[0]
			del self.gt_yaw[0]
			del self.odom_XY[0]
			del self.odom_yaw[0]

			# calculate distance traveled and yaw difference
			gt_dist = math.sqrt((self.gt_XY[self.diff_duration-1][0] - self.gt_XY[0][0])**2 + (self.gt_XY[self.diff_duration-1][1] - self.gt_XY[0][1])**2)
			gt_yaw = self.gt_yaw[self.diff_duration-1] - self.gt_yaw[0]
			odom_dist = math.sqrt((self.odom_XY[self.diff_duration-1][0] - self.odom_XY[0][0])**2 + (self.odom_XY[self.diff_duration-1][1] - self.odom_XY[0][1])**2)
			odom_yaw = self.odom_yaw[self.diff_duration-1] - self.odom_yaw[0]	

			self.dist_diff_instant = odom_dist - gt_dist
			self.yaw_diff_instant = odom_yaw - gt_yaw 

			if(self.yaw_diff_instant <= -2*math.pi):
				self.yaw_diff_instant = -self.yaw_diff_instant
			
			self.dist_diff.append(self.dist_diff_instant)
			self.yaw_diff.append(self.yaw_diff_instant)
			self.vels2.append([odom_data.twist.twist.linear.x, odom_data.twist.twist.angular.z])
			# print("Length of the odom error vector = ", len(self.dist_diff))

			# if (len(self.dist_diff) == 1000):
			# 	print("Saving Files!")
			# 	np.save("/home/adarshjs/numpy_files/diff_duration_12/" + self.surface_name + "_vel_diff_12.npy", self.dist_diff)
			# 	np.save("/home/adarshjs/numpy_files/diff_duration_12/" + self.surface_name + "_yaw_rate_diff_12.npy", self.yaw_diff)
			# 	np.save("/home/adarshjs/numpy_files/diff_duration_12/" + self.surface_name + "_vel", self.vels2)
			# 	quit()		


	def PCA(self, X , num_components):
		#Step-1
		X_meaned = X - np.mean(X , axis = 0)

		#Step-2
		cov_mat = np.cov(X_meaned , rowvar = False)

		#Step-3
		eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

		#Step-4
		sorted_index = np.argsort(eigen_values)[::-1]
		sorted_eigenvalue = eigen_values[sorted_index]
		sorted_eigenvectors = eigen_vectors[:,sorted_index]

		#Step-5
		eigenvector_subset = sorted_eigenvectors[:,0:num_components]

		#Step-6
		X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()

		return X_reduced
		

def main(args):
	ic = Dataset_Subscriber()
	rospy.init_node('dataset_subscriber', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
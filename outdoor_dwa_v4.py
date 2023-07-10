#!/usr/bin/env python
from __future__ import print_function

# Author: Connor McGuile
# Feel free to use in any way.

# A custom Dynamic Window Approach implementation for use with Turtlebot.
# Obstacles are registered by a front-mounted laser and stored in a set.
# If, for testing purposes or otherwise, you do not want the laser to be used,
# disable the laserscan subscriber and create your own obstacle set in main(),
# before beginning the loop. If you do not want obstacles, create an empty set.
# Implentation based off Fox et al.'s paper, The Dynamic Window Approach to
# Collision Avoidance (1997).

# This code integrates our self-supervised learning output costmap with DWA by computing the surface cost for each v, w in the Dynamic Window
# TODO: Varying the acceleration

import roslib
import rospy
import math
import numpy as np
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
import time
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
import sys
import csv
# from odom_calculator import odom_calulator

# Network related imports
from model_bn import get_model
from tqdm import tqdm
from PIL import Image as im
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import scipy.misc


class Config():

    def __init__(self):
        # robot parameter
        #NOTE good params:
        #NOTE 0.55,0.1,1.0,1.6,3.2,0.15,0.05,0.1,1.7,2.4,0.1,3.2,0.18
        # self.bridge = CvBridge()
        self.max_speed = 0.5#0.6  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yawrate = 0.5#0.6  # [rad/s]
        self.max_accel = 2.5  # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        self.v_reso = 0.05 # [m/s]
        self.yawrate_reso = 0.1  # [rad/s]
        self.dt = 0.5  # [s]
        self.predict_time = 1.5  # [s]
        self.time_reso = 0.02
        self.upper_cost_threshold = 0.25

        self.vel_limit = self.max_speed
        self.yawrate_limit = self.max_yawrate
        
        self.to_goal_cost_gain = 1.5#2.4 #lower = detour
        self.speed_cost_gain = 0.1 #lower = faster
        self.obs_cost_gain = 3.2 #lower = fearless
        self.robot_radius = 0.5  # [m]
        self.surface_cost_gain = 50 # lower = cost difference between the surfaces will be low
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.goalX = 0
        self.goalY = 0
        self.th = 0.0
        self.r = rospy.Rate(20)

        # self.goalX = 2.0
        # self.goalY = 0

        # List lengths related to inputs and labels 
        self.patch_side = 100
        self.cropping_row = 150 #76
        self.num_h = 7
        self.num_v = 2
        self.stride = 100
        self.vel_vector_len = 50
        self.diff_duration = 12
        self.iter = 0

        self.cropped_rows=0
        self.cropped_cols=0

        # Image object
        self.resized_img = np.zeros((self.num_v*100, self.num_h*100, 3), np.uint8)
        self.cropped_img = np.zeros((330, 640, 3), np.uint8) #np.zeros((300, 672, 3), np.uint8) 
        self.input_imgs = np.asarray([])

        # Input velocity vector
        self.vels = []
        self.input_vel = np.asarray([])
        self.vel_array_reshaped = np.asarray([])

        self.cropped_list = []
        self.divided_patch_list = []

        # Load trained model
        self.model_lst = "Weights-047--0.40830.hdf5"
        self.model = load_model(self.model_lst)
        print("Finished Loading Model!")

        # Topic names
        self.odom_topic_name = "/odometry/filtered"
        self.image_topic_name="/camera/color/image_raw"
        # self.image_topic_name = "/zed2i/zed_node/left_raw/image_raw_color"

        self.camera_tilt_angle = -30


    # Callback for Odometry
    def assignOdomCoords(self, msg):

        # X- and Y- coords and pose of robot fed back into the robot config
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        rot_q = msg.pose.pose.orientation
        (roll,pitch,theta) = euler_from_quaternion (rot_q.x,rot_q.y,rot_q.z,rot_q.w) # for odometry
        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.z, -rot_q.x, -rot_q.y, rot_q.w]) # for LeGO LOAM topic
        self.th = theta

    # def dist_vibration_calculator(self):
    #     # send odom data to planner to calculate total trajectory distance 
    #     #and robot vibration using z
    #     end_goal_global_xy = [self.goalX,self.goalY]
    #     current_z = self.z
    #     current_x = self.x
    #     current_y = self.y
        
    #     return current_x,current_y,current_z,end_goal_global_xy


    def img_callback(self, img_data):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
            cv_input = np.frombuffer(img_data.data, dtype=np.uint8).reshape(img_data.height, img_data.width, -1)


        except CvBridgeError as e:
            print(e)

        # print("Inside Image Callback")
        # Obtain patch
        cv_image = cv_input[:,:,0:3] # for zed camera only
        (rows,cols,channels) = cv_image.shape
        # print("Original image size:", (rows,cols,channels))
        # cropped_image = cv_image[int(rows-self.patch_side):int(rows), int(cols/2 - self.patch_side/2):int(cols/2 + self.patch_side/2)]
        self.cropped_img = cv_image[self.cropping_row:int(rows), 0:int(cols)]
        (self.cropped_rows,self.cropped_cols,chls) = np.shape(self.cropped_img)

        resize_width = (self.num_h*self.patch_side) - ((self.num_h-1)*(self.patch_side-self.stride))
        resize_height = (self.num_v*self.patch_side) - ((self.num_v-1)*(self.patch_side-self.stride))
        dim = (resize_width, resize_height)
  
        # resize image
        self.resized_img = cv2.resize(self.cropped_img, dim, interpolation = cv2.INTER_AREA)





    # Callback for goal from POZYX
    def target_callback(self, data):

        radius = data.linear.x # this will be r
        theta = data.linear.y * 0.0174533 # this will be theta
        
        # Goal coordinate wrt robot frame
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        # Converting goal coordinate to world frame (where robot started)
        self.goalX =  self.x + goalX_rob*math.cos(self.th) - goalY_rob*math.sin(self.th)
        self.goalY = self.y + goalX_rob*math.sin(self.th) + goalY_rob*math.cos(self.th)
        print("Self odom:",self.x,self.y)
        print("Goals = ", self.goalX, self.goalY)


    def weak_segmentation(self, cropped_image):
        #performs weak classification on the resized cropped image

        gray_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        ret, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
        segmentation =binary_img

        # img_channel=cropped_image[:,0,:]
        # # print("img channel",img_channel)
        # pixel_intensities = img_channel.flatten()

        # sobel_map = sobel(img_channel)
        # markers = np.zeros_like(img_channel)

        ## add marker estimation from GMM

        # thresholds= self.GMM_marker_thresholds(pixel_intensities)

        # for th in range(len(thresholds)):
        #   if th == 0:
        #       markers[img_channel <= thresholds[th]] = th
        #   else: #th < len(thresholds):
        #       markers[thresholds[th-1]< img_channel <= thresholds[th]] = th
                
        # markers[thresholds[len(thresholds)-1] > img_channel] = len(thresholds)


        # markers[img_channel <=150] = 1
        # markers[img_channel > 150] = 0

        # seg = skimage.segmentation.watershed(sobel_map, markers)

        # segmentation = im.fromarray(np.array(seg))

        # # segmentation = ndi.binary_fill_holes(segmentation - 1)

        # # labels, _ = ndi.label(segmentation)
        # # image_label_overlay = label2rgb(labels, image=imm)
        # # print("segmentation:",segmentation)

        return segmentation


    def traj_cost_calculation(self,image, x_d, theta_d):
        #takes patch based image as input and calculate cost values for a given set of trajectories (v,w pairs)
        # cost_list =[]

        resized_cost_image = cv2.resize(image, (self.cropped_cols,self.cropped_rows), interpolation = cv2.INTER_AREA)
        # print("Resized cost image shape:",np.shape(resized_cost_image))

        x_t =[]
        y_t =[]

        time_steps = np.arange(0.1, self.predict_time, self.time_reso) 
        for i in time_steps:
            # we put -ve here to follow DWA's convention of left +ve. This function normally follows right +ve
            y_t.append((x_d * np.cos(-theta_d*(i)))*(i)) 
            x_t.append((x_d * np.sin(-theta_d*(i)))*(i))
            
        height =0.3
        h_vec = np.ones(len(x_t))* height
        points = np.transpose([x_t,h_vec, y_t])

        # print("Trajectroy Points",points)
        # print(np.shape(points))

        # # Transform x,y,z ground coordinates to camera frame
        alpha = np.deg2rad(self.camera_tilt_angle)
        Rotation_mat = [[1, 0             , 0],
                        [0, np.cos(alpha), -np.sin(alpha)],
                        [0, np.sin(alpha), np.cos(alpha)]]

        points_rotated = np.matmul(points,Rotation_mat)

        X0 = np.ones((points_rotated.shape[0],1))
        pointsnew = np.hstack((points_rotated,X0))

        # Projection/camera matrix
        # P = [[613.6345825195312, 0.0, 314.1249084472656, 0.0], [0.0, 613.6775512695312, 246.6942138671875, 0.0], [0.0, 0.0, 1.0, 0.0]] # realsense 435
        P = [[607.175048828125, 0.0, 322.55340576171875, 0.0], [0.0, 607.222900390625, 248.86021423339844, 0.0], [0.0, 0.0, 1.0, 0.0]] # realsense lidar camera L515
        # P = [[266.9125061035156, 0.0, 336.4200134277344, 0.0], [0.0, 266.94500732421875, 181.58975219726562, 0.0], [0.0, 0.0, 1.0, 0.0]] #zed camera left
        # P= [[273.0054626464844, 0.0, 342.40814208984375, -32.74513626098633], [0.0, 273.0054626464844, 185.3704833984375, 0.0], [0.0, 0.0, 1.0, 0.0]] #zed camera right

        uvw = np.dot(P,np.transpose(pointsnew))

        u_vec = uvw[0]
        v_vec = uvw[1]
        w_vec = uvw[2]

        x_vec = u_vec / w_vec
        y_vec = v_vec / w_vec

        # remove pixel coordinates that are out of the image range
        # print("rows, cols:",self.cropped_rows,self.cropped_cols)

        out_of_range_x = [index for index,value in enumerate(x_vec) if  value > self.cropped_cols]
        out_of_range_x2 = [index for index,value in enumerate(x_vec) if  value < 0]
        out_of_range_y = [index for index,value in enumerate(y_vec) if  value > self.cropped_rows]
        out_of_range_y2 = [index for index,value in enumerate(y_vec) if  value < 0]
        out_of_range_list = list(set(out_of_range_x) | set(out_of_range_y)|set(out_of_range_x2) | set(out_of_range_y2))
        # print("out_of_range_list",out_of_range_list)

        def merge(x_vec, y_vec):        
            merged_list = [(int(x_vec[i]), int(y_vec[i])) for i in range(0, len(x_vec))]
            return merged_list

        imagepoints_vec= np.array(merge(x_vec, y_vec))
        # imagepoints = imagepoints_vec

        imagepoints = np.delete(imagepoints_vec,out_of_range_list,axis=0)
        # print("Final # imagepoints:",imagepoints)

        if len(imagepoints) > 0:
            cost =0
            rows,cols = imagepoints.shape
            for i in range(len(imagepoints)):
                k = resized_cost_image[imagepoints[i][1],imagepoints[i][0]]
                cost = cost + k
                # print("Traj. Cost:", cost)
                out_image = cv2.polylines(resized_cost_image,[imagepoints],isClosed=False,color=(0, 255, 0),thickness=3,lineType=cv2.LINE_AA)

            cost_norm = cost #/len(imagepoints)
        
        else:
            # print("No trajectory exists for this v,w pair")
            out_image = resized_cost_image

            # #for lidar camera
            if x_d == 0 and theta_d > 0: # following DWA's convention (left is +ve and right is -ve )
                cost_norm = resized_cost_image[329, 10] # cost of the bottom left patch
            elif x_d == 0 and theta_d < 0:
                cost_norm = resized_cost_image[329, 630] # cost of the bottom right patch
            elif x_d != 0 and theta_d != 0:
                cost_norm = 1    # This was a random choice
            else:
                cost_norm = 0 # (v, w) = (0, 0)

            # cost_norm = 10

            # # For ZED
            # if x_d == 0 and theta_d > 0: # following DWA's convention (left is +ve and right is -ve )
            #     cost_norm = resized_cost_image[290, 10] # cost of the bottom left patch
            # elif x_d == 0 and theta_d < 0:
            #     cost_norm = resized_cost_image[290, 650] # cost of the bottom right patch
            # elif x_d != 0 and theta_d != 0:
            #     cost_norm = 1    # This was a random choice
            # else:
            #     cost_norm = 0 # (v, w) = (0, 0)


        return out_image, cost_norm

    def traj_cost_calculation2(self,image):
        #takes patch based image as input and calculate cost values for a given set of trajectories (v,w pairs)
        cost_list =[]

        resized_cost_image = cv2.resize(image, (self.cropped_cols,self.cropped_rows), interpolation = cv2.INTER_AREA)
        # print("Resized cost image shape:",np.shape(resized_cost_image))

        linear_vels = np.arange(self.min_speed,self.max_speed+self.v_reso,self.v_reso)
        angular_vels = np.arange(-self.max_yawrate,self.max_yawrate+self.yawrate_reso,self.yawrate_reso)
        for x_d in linear_vels:
          for theta_d in angular_vels:
            # x_d = 0.5 #data.linear.x
            # theta_d = 0.1 #data.angular.z   #* (180/np.pi)

            # print("Linear and angular vels:",(x_d, theta_d))

            x_t =[]
            y_t =[]

            time_steps = np.arange(0.1, self.predict_time, self.time_reso) 
            for i in time_steps:
                y_t.append((x_d * np.cos(theta_d*(i)))*(i))
                x_t.append((x_d * np.sin(theta_d*(i)))*(i))
                
            height =0.3
            h_vec = np.ones(len(x_t))* height
            points = np.transpose([x_t,h_vec, y_t])

            # print("Trajectroy Points",points)
            # print(np.shape(points))

            # # Transform x,y,z ground coordinates to camera frame
            alpha = np.deg2rad(self.camera_tilt_angle)
            Rotation_mat = [[1, 0             , 0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]]

            points_rotated = np.matmul(points,Rotation_mat)

            X0 = np.ones((points_rotated.shape[0],1))
            pointsnew = np.hstack((points_rotated,X0))

            # Projection/camera matrix
            # P = [[613.6345825195312, 0.0, 314.1249084472656, 0.0], [0.0, 613.6775512695312, 246.6942138671875, 0.0], [0.0, 0.0, 1.0, 0.0]] # realsense 435
            P = [[607.175048828125, 0.0, 322.55340576171875, 0.0], [0.0, 607.222900390625, 248.86021423339844, 0.0], [0.0, 0.0, 1.0, 0.0]] # realsense lidar camera L515
            # P = [[266.9125061035156, 0.0, 336.4200134277344, 0.0], [0.0, 266.94500732421875, 181.58975219726562, 0.0], [0.0, 0.0, 1.0, 0.0]] #zed camera left
            # P= [[273.0054626464844, 0.0, 342.40814208984375, -32.74513626098633], [0.0, 273.0054626464844, 185.3704833984375, 0.0], [0.0, 0.0, 1.0, 0.0]] #zed camera right

            uvw = np.dot(P,np.transpose(pointsnew))

            u_vec = uvw[0]
            v_vec = uvw[1]
            w_vec = uvw[2]

            x_vec = u_vec / w_vec
            y_vec = v_vec / w_vec

            # remove pixel coordinates that are out of the image range
            # print("rows, cols:",self.cropped_rows,self.cropped_cols)

            out_of_range_x = [index for index,value in enumerate(x_vec) if  value > self.cropped_cols]
            out_of_range_x2 = [index for index,value in enumerate(x_vec) if  value < 0]
            out_of_range_y = [index for index,value in enumerate(y_vec) if  value > self.cropped_rows]
            out_of_range_y2 = [index for index,value in enumerate(y_vec) if  value < 0]
            out_of_range_list = list(set(out_of_range_x) | set(out_of_range_y)|set(out_of_range_x2) | set(out_of_range_y2))
            # print("out_of_range_list",out_of_range_list)

            def merge(x_vec, y_vec):        
                merged_list = [(int(x_vec[i]), int(y_vec[i])) for i in range(0, len(x_vec))]
                return merged_list

            imagepoints_vec= np.array(merge(x_vec, y_vec))
            # imagepoints = imagepoints_vec

            imagepoints = np.delete(imagepoints_vec,out_of_range_list,axis=0)
            # print("Final # imagepoints:",imagepoints)

            if len(imagepoints) > 0:
                cost =0
                rows,cols = imagepoints.shape
                for i in range(len(imagepoints)):
                    k = resized_cost_image[imagepoints[i][1],imagepoints[i][0]]
                    cost = cost + k
                    # print("Traj. Cost:", cost)
                    out_image = cv2.polylines(resized_cost_image,[imagepoints],isClosed=False,color=(0, 255, 0),thickness=3,lineType=cv2.LINE_AA)

                cost_list.append(cost/len(imagepoints))
            else:
                # print("No trajectory exists for this v,w pair")
                out_image = resized_cost_image
                cost_list.append(np.inf) # set traj cost to inf since no points exists for that v,w pair
        # cv2_imshow(imag)

        return out_image, cost_list




def divide_patch(patch):
    subpatch_1 = patch[0:int(patch.shape[0]/2), 0:int(patch.shape[0]/2)]
    subpatch_2 = patch[0:int(patch.shape[0]/2), int(patch.shape[0]/2):patch.shape[0]]
    subpatch_3 = patch[int(patch.shape[0]/2):patch.shape[0], 0:int(patch.shape[0]/2)]
    subpatch_4 = patch[int(patch.shape[0]/2):patch.shape[0], int(patch.shape[0]/2):patch.shape[0]]
    
    # Sanity check for input image
    # hori1 = np.concatenate((subpatch_1, subpatch_2), axis=1)
    # hori2 = np.concatenate((subpatch_3, subpatch_4), axis=1)
    # full_img = np.concatenate((hori1, hori2), axis=0)
    # cv2.imshow("Divided Patch Put Together", full_img)
    # cv2.waitKey(100)

    # Resize subpatches to 100x100 size to feed as input to network
    resized_sp_1 = cv2.resize(subpatch_1, (100, 100), interpolation = cv2.INTER_AREA)
    resized_sp_2 = cv2.resize(subpatch_2, (100, 100), interpolation = cv2.INTER_AREA)
    resized_sp_3 = cv2.resize(subpatch_3, (100, 100), interpolation = cv2.INTER_AREA)
    resized_sp_4 = cv2.resize(subpatch_4, (100, 100), interpolation = cv2.INTER_AREA)

    return resized_sp_1, resized_sp_2, resized_sp_3, resized_sp_4


def put_together(cost1, cost2, cost3, cost4, patch_side):
    sp1 = np.full((int(patch_side/2), int(patch_side/2)), cost1)
    sp2 = np.full((int(patch_side/2), int(patch_side/2)), cost2)
    sp3 = np.full((int(patch_side/2), int(patch_side/2)), cost3)
    sp4 = np.full((int(patch_side/2), int(patch_side/2)), cost4)

    hori1 = np.concatenate((sp1, sp2), axis=1)
    hori2 = np.concatenate((sp3, sp4), axis=1)
    full_img = np.concatenate((hori1, hori2), axis=0)

    return full_img

time_list = []
class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in view
        self.obst = set()
        self.collision_status = False


    # Custom range implementation to loop over LaserScan degrees with
    # a step and include the final degree
    def myRange(self,start,end,step):
        i = start
        while i < end:
            yield i
            i += step
        yield end


    def return_time_taken(self):
        return self.time_list


    # Callback for LaserScan
    def assignObs(self, msg, config):

        deg = len(msg.ranges)   # Number of degrees - varies in Sim vs real world
        # print("Laser degree length {}".format(deg))
        self.obst = set()   # reset the obstacle set to only keep visible objects

        maxAngle = 360
        scanSkip = 1
        anglePerSlot = (float(maxAngle) / deg) * scanSkip
        angleCount = 0
        angleValuePos = 0
        angleValueNeg = 0
        
        self.collision_status = False
        for angle in self.myRange(0,deg-1,scanSkip):
            distance = msg.ranges[angle]

            if (distance < 0.05) and (not self.collision_status):
                self.collision_status = True
                # print("Collided")
                # reached = False
                

            if(angleCount < (deg / (2*scanSkip))):
                # print("In negative angle zone")
                angleValueNeg += (anglePerSlot)
                scanTheta = (angleValueNeg - 180) * math.pi/180.0


            elif(angleCount>(deg / (2*scanSkip))):
                # print("In positive angle zone")
                angleValuePos += anglePerSlot
                scanTheta = angleValuePos * math.pi/180.0
            # only record obstacles that are within 4 metres away

            else:
                scanTheta = 0

            angleCount += 1

            if (distance < 4):

                objTheta =  scanTheta + config.th
                # round coords to nearest 0.125m
                obsX = round((config.x + (distance * math.cos(abs(objTheta))))*8)/8
                # determine direction of Y coord
                # if (objTheta < 0): # uncomment and comment line below for Gazebo simulation
                if (objTheta < 0):
                    obsY = round((config.y - (distance * math.sin(abs(objTheta))))*8)/8
                else:
                    obsY = round((config.y + (distance * math.sin(abs(objTheta))))*8)/8

                # print("Robot's current location {} {}".format(config.x, config.y))
                # print("Obstacle's current location {} {}".format(obsX, obsY))
                # print("Current yaw of the robot {}".format(config.th))

                # add coords to set so as to only take unique obstacles
                self.obst.add((obsX,obsY))
                # print("The obstacle space is {}".format(self.obst))
                #print self.obst
        # print("The total angle count is {}".format(angleCount  ))


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians  (counterclockwise)
    yaw is rotation around z in radians  (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians


# Model to determine the expected position of the robot after moving along trajectory
def motion(x, u, dt):
    # Motion model
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    # u = [v, w]

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


# Determine the dynamic window from robot configurations
def calc_dynamic_window(x, config,cost_map):

    max_cost = 2 # max cost of the actual cost map
    print("Max cost of the costmap:",np.max(np.max(cost_map)))
    cost_map_normalized = (cost_map/max_cost)* (math.pi/2)
    # # calculate velocity limits
    avg_cost = np.mean(cost_map_normalized)
    print("Average cost:",avg_cost)

    accel_factor = math.cos(avg_cost)
    # print("Spped factor:",speed_factor) 

    max_accel_limit = config.max_accel * accel_factor
    max_dyawrate_limit = config.max_dyawrate * accel_factor

    vel_lower_limit = config.vel_limit * accel_factor
    print("vel limit:",config.vel_limit)

    if avg_cost >= config.upper_cost_threshold and config.vel_limit > 0.35:
        print("Bad surface vel limit:",config.vel_limit)
        config.vel_limit = config.vel_limit - config.max_accel*0.05
        # config.yawrate_limit = config.yawrate_limit - config.max_dyawrate*0.05
    elif avg_cost < config.upper_cost_threshold and config.vel_limit < config.max_speed:
        print("Good surface vel limit:",config.vel_limit)
        config.vel_limit = config.vel_limit + max_accel_limit*0.05
        # config.yawrate_limit = config.yawrate_limit + max_dyawrate_limit*0.05
    elif config.vel_limit < 0.35 :
        config.vel_limit = 0.35
        # config.yawrate_limit = 0.4

    # Full search space
    Vs = [config.min_speed, config.vel_limit,
          -config.yawrate_limit, config.yawrate_limit]

    # # Full search space
    # Vs = [config.min_speed, config.max_speed,
    #       -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model

    # CALL FUNCTION TO COMPUTE ACCELERATION LIMITS HERE

    Vd = [x[3] - config.max_accel * config.dt,
    x[3] + max_accel_limit * config.dt,
    x[4] - max_dyawrate_limit * config.dt,
    x[4] + max_dyawrate_limit * config.dt]


    # TODO: CHANGE LINEAR AND ANGULAR ACCELERATION LIMITS (DEFINE IN CONFIG CONSTRUCTOR)
    # Vd = [x[3] - config.max_accel * config.dt,
    #       x[3] + config.max_accel * config.dt,
    #       x[4] - config.max_dyawrate * config.dt,
    #       x[4] + config.max_dyawrate * config.dt]


    #  [vmin, vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


# Calculate a trajectory sampled across a prediction time
def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0

    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt) # next state
        traj = np.vstack((traj, x)) # store each state along a trajectory
        time += config.dt # next sample

    return traj


# Calculate trajectory, costings, and return velocities to apply to robot
def calc_final_input(x, u, dw, config, ob, cost_map):

    xinit = x[:]
    min_cost = 10000.0
    min_u = u
    min_u[0] = 0.0

    # evaluate all trajectory with sampled input in dynamic window
    print("=======================================================")
    for v in np.arange(dw[0], dw[1], config.v_reso):
        for w in np.arange(dw[2], dw[3], config.yawrate_reso):
            traj = calc_trajectory(xinit, v, w, config)

            # calc costs with weighted gains
            to_goal_cost = calc_to_goal_cost(traj, config) * config.to_goal_cost_gain
            
            speed_cost = config.speed_cost_gain * \
                (config.max_speed - traj[-1, 3])

            ob_cost = calc_obstacle_cost(traj, ob, config) * config.obs_cost_gain

            # TODO: COMPUTE SURFACE COST HERE FOR V, W
            _ , sur_cost = config.traj_cost_calculation(cost_map, v, w)
            sur_cost = sur_cost*config.surface_cost_gain
            

            # final_cost = to_goal_cost + speed_cost + ob_cost + 5*sur_cost # TUNE
            # final_cost = to_goal_cost + ob_cost + 10*sur_cost # without speed cost
            # final_cost = speed_cost + ob_cost + 10*sur_cost
            final_cost = ob_cost + to_goal_cost*(1+sur_cost) #+(sur_cost/config.surface_cost_gain)


            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                min_u = [v, w]
                
    print("(v, w) = ", min_u)
    traj = calc_trajectory(xinit, min_u[0], min_u[1], config)
    to_goal_cost = calc_to_goal_cost(traj, config) * config.to_goal_cost_gain
    _ , sur_cost = config.traj_cost_calculation(cost_map, min_u[0], min_u[1])
    sur_cost = sur_cost*config.surface_cost_gain
    print("Goal:{:.2f}, Sur:{:.2f}".format(to_goal_cost, sur_cost) )
    # print("Goal:{:.2f}, vel:{:.2f}, Obs:{:.2f}, Sur:{:.2f}".format(to_goal_cost, speed_cost, ob_cost, sur_cost) )
    return min_u


# Calculate obstacle cost inf: collision, 0:free
def calc_obstacle_cost(traj, ob, config):
    skip_n = 2
    minr = float("inf")

    # Loop through every obstacle in set and calc Pythagorean distance
    # Use robot radius to determine if collision
    for ii in range(0, len(traj[:, 1]), skip_n):
        for i in ob.copy():
            ox = i[0]
            oy = i[1]
            dx = traj[ii, 0] - ox
            dy = traj[ii, 1] - oy

            r = math.sqrt(dx**2 + dy**2)

            if r <= config.robot_radius:
                return float("Inf")  # collision

            if minr >= r:
                minr = r

    return 1.0 / minr


# Calculate goal cost via Pythagorean distance to robot
def calc_to_goal_cost(traj, config):
    # If-Statements to determine negative vs positive goal/trajectory position
    # traj[-1,0] is the last predicted X coord position on the trajectory
    if (config.goalX >= 0 and traj[-1,0] < 0):
        dx = config.goalX - traj[-1,0]
    elif (config.goalX < 0 and traj[-1,0] >= 0):
        dx = traj[-1,0] - config.goalX
    else:
        dx = abs(config.goalX - traj[-1,0])
    # traj[-1,1] is the last predicted Y coord position on the trajectory
    if (config.goalY >= 0 and traj[-1,1] < 0):
        dy = config.goalY - traj[-1,1]
    elif (config.goalY < 0 and traj[-1,1] >= 0):
        dy = traj[-1,1] - config.goalY
    else:
        dy = abs(config.goalY - traj[-1,1])

    cost = math.sqrt(dx**2 + dy**2)
    return cost



# Begin DWA calculations
def dwa_control(x, u, config, ob, cost_map):
    # Dynamic Window control
    dw = calc_dynamic_window(x, config,cost_map)
    u = calc_final_input(x, u, dw, config, ob, cost_map)
    return u


# Determine whether the robot has reached its goal
def atGoal(config, x,goal_state_pub):
    # check at goal
    goal_reached = Bool() 
    goal_reached.data = False
    if math.sqrt((x[0] - config.goalX)**2 + (x[1] - config.goalY)**2) \
        <= config.robot_radius:
        goal_reached.data = True
        goal_state_pub.publish(goal_reached) 
        return True
    goal_state_pub.publish(goal_reached)    
    return False


def main():
    print(__file__ + " start!!")
    
    config = Config() # robot specification
    obs = Obstacles() # position of obstacles

    # subOdom = rospy.Subscriber("/integrated_to_init", Odometry, config.assignOdomCoords)
    subOdom = rospy.Subscriber(config.odom_topic_name, Odometry, config.assignOdomCoords)
    subLaser = rospy.Subscriber("/scan", LaserScan, obs.assignObs, config)
    subGoal = rospy.Subscriber('/target/position', Twist, config.target_callback)
    subImage = rospy.Subscriber(config.image_topic_name, Image, config.img_callback)
    goal_state_pub = rospy.Publisher('/goal_state_pub', Bool, queue_size=10)

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    # pub = rospy.Publisher('/husky/cmd_vel_DWA', Twist, queue_size=1)
    speed = Twist()
    
    # initial state [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x = np.array([config.x, config.y, config.th, 0.0, 0.0])
    # initial linear and angular velocities
    u = np.array([0.0, 0.0])

    reached = False
    reached_goal = 0


    # runs until terminated externally
    while not rospy.is_shutdown():

        if (atGoal(config,x,goal_state_pub) == False): # Not reached the goal
            t1 = time.time()

            # patch generation from the weak classifier
            segmented_img = config.weak_segmentation(config.resized_img)
            t2 = time.time()
            config.cropped_list = []
            out_patches = []
            patch_counter = 0
            config.divided_patch_list = []

            # Cropping the RGB patches and creating a list
            for j in range(0, config.resized_img.shape[0]- config.stride+1, config.stride):
                for i in range(0, config.resized_img.shape[1]-config.stride+1, config.stride):
                    # config.cropped_list.append(resized_img[j:j + config.patch_side, i:i + config.patch_side])

                    # Condition based on number of white pixels and black pixels (TODO)
                    if(np.count_nonzero(segmented_img[j:j + config.patch_side, i:i + config.patch_side]) < 0.5*config.patch_side*config.patch_side): 
                        sp1, sp2, sp3, sp4 = divide_patch(config.resized_img[j:j + config.patch_side, i:i + config.patch_side])
                        config.cropped_list.append(sp1)
                        config.cropped_list.append(sp2)
                        config.cropped_list.append(sp3)
                        config.cropped_list.append(sp4)
                        config.divided_patch_list.append(patch_counter)
                        patch_counter = patch_counter + 4
                        # print("Patch number = ", patch_counter)

                    else:
                        config.cropped_list.append(config.resized_img[j:j + config.patch_side, i:i + config.patch_side])
                        patch_counter = patch_counter + 1

            
            config.input_imgs = np.asarray(config.cropped_list)
            
            # Obtain Odom and append on to a vector
            odom_data = rospy.wait_for_message(config.odom_topic_name, Odometry, timeout=None)
            config.vels.append([odom_data.twist.twist.linear.x, odom_data.twist.twist.angular.z])
            
            if len(config.vels) > config.vel_vector_len:
                # Delete the oldest velocity
                config.vels.pop(0)
                vel_array = np.asarray(config.vels)
                vel_array_reshaped = vel_array.reshape(-1)
                # print("Reshaped velocity vector shape: ", vel_array_reshaped.shape)
                
                config.input_vel = np.tile(vel_array_reshaped,(len(config.cropped_list),1))
                # print("Input velocity vector shape: ", input_vel.shape)

            # if len(config.vels) == config.vel_vector_len and config.input_imgs.shape[0] == config.input_vel.shape[0]:
                
                print("Input images shape", config.input_imgs.shape)
                # print("Input velocities shape", config.input_vel.shape)
                # t1 = time.time()
                # Inference
                out = config.model.predict([config.input_imgs, config.input_vel])
                # t2 = time.time()
                out_patches=[]
        
                k = 0 # iterates over all the outputs
                itr = 0 # iterates between 0-13 (number of 100x100 patches)

                while k in range(len(np.linalg.norm(out, axis=1))):
                    if(k in config.divided_patch_list):
                        # print("If condition satisfied")
                        put_together_img = put_together(np.linalg.norm(out, axis=1)[k], np.linalg.norm(out, axis=1)[k+1], np.linalg.norm(out, axis=1)[k+2], np.linalg.norm(out, axis=1)[k+3], 100)
                        out_patches.append(put_together_img)
                        k = k + 4

                    else:    
                        out_patches.append(np.full((config.patch_side, config.patch_side), np.linalg.norm(out, axis=1)[k]))
                        k = k + 1

                    # print("k = ", k)

                # for i in range(len(np.linalg.norm(out, axis=1))):
                #   out_patches.append(np.full((config.patch_side, config.patch_side), np.linalg.norm(out, axis=1)[i]))

                out_patches_array = np.asarray(out_patches)
                # # print("Norm vec:",out_patches_array)
                # print("Norm vec size",out_patches_array.shape)

                hori1 = np.concatenate((out_patches_array[0], out_patches_array[1], out_patches_array[2], out_patches_array[3], out_patches_array[4], out_patches_array[5], out_patches_array[6]), axis=1)
                hori2 = np.concatenate((out_patches_array[7], out_patches_array[8], out_patches_array[9], out_patches_array[10], out_patches_array[11], out_patches_array[12], out_patches_array[13]), axis=1)
                cost_map = np.concatenate((hori1, hori2), axis=0)

                # calculate trajectory costs for a set of v,w on the cost image
                # out_img, _ = config.traj_cost_calculation2(cost_map)
                # print("Trajectory cost list:",traj_costs)

                
                print("Inference Time:",t2-t1)
                print("___________________________")

                # cv2.imshow("Input image", config.cropped_img)
                # cv2.imshow("Prediction", cost_map)
                cv2.waitKey(1)

                u = dwa_control(x, u, config, obs.obst, cost_map)
                x[0] = config.x
                x[1] = config.y
                x[2] = config.th
                x[3] = u[0]
                x[4] = u[1]
                speed.linear.x = x[3]
                speed.angular.z = x[4]
                
        else:
            # if at goal then stay there until new goal published
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0])
            # print("Else condition satisfied. ")
        
        pub.publish(speed)
        config.r.sleep()

        

        # print("Goals = ", config.goalX, config.goalY)
        # print("Current Position = ", x[0], x[1])

        



if __name__ == '__main__':
    rospy.init_node('outdoor_dwa')

    global initial_pose, robot_goal
    initial_pose = {}
    robot_goal = {}

    initial_pose["x_init"] = 0
    initial_pose["y_init"] = 0
    initial_pose["x_rot_init"] = 0
    initial_pose["y_rot_init"] = 0
    initial_pose["z_rot_init"] = 0
    initial_pose["w_rot_init"] = 1
    robot_goal["x"] = 0
    robot_goal["y"] = 0

    main()

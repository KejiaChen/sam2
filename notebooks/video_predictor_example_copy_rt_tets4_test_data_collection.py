#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/kifabrik/Documents/segment-anything-2')
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# import open3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import pyrealsense2 as rs
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters
from std_msgs.msg import Float64MultiArray
import plotly.graph_objs as go
from sklearn.cluster import DBSCAN
from UDPReceiver import UDPReceiver
from scipy.optimize import curve_fit
import socket
import json
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from scipy.ndimage import median_filter
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from sklearn.neighbors import NearestNeighbors
import torch.jit
from sklearn.neighbors import BallTree
import pandas as pd
from scipy.interpolate import splprep, splev
from matplotlib.animation import FuncAnimation
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch
import argparse

def fit_b_spline(points, num_points=10):
    tck, _ = splprep(points.T, s=0)  # Compute B-spline representation
    u_new = np.linspace(0, 1, num_points)  # Generate uniformly spaced parameter values
    spline_points = np.array(splev(u_new, tck)).T  # Generate B-spline points
    return spline_points

def fit_polynomial(points, degree=5, num_points=100):
    # Fit a polynomial curve to the points
    y = points[:, 1]
    x = points[:, 0]
    z = points[:, 2]

    # Fit polynomials to x and z as functions of y
    poly_x = np.polyfit(y, x, degree)
    poly_z = np.polyfit(y, z, degree)

    # Generate evenly spaced y values for interpolation
    y_interp = np.linspace(min(y), max(y), num_points)

    # Calculate x and z values using the polynomial coefficients
    x_interp = np.polyval(poly_x, y_interp)
    z_interp = np.polyval(poly_z, y_interp)

    return np.column_stack((x_interp, y_interp, z_interp))

def fit_bezier(points, num_points=100, max_degree=10):
    # Limit the degree to avoid overflow errors
    n = min(len(points) - 1, max_degree)
    
    if len(points) - 1 > max_degree:
        # Downsample points if they exceed max_degree + 1
        indices = np.linspace(0, len(points) - 1, max_degree + 1).astype(int)
        points = points[indices]

    bezier_curve = np.zeros((num_points, 3))
    t = np.linspace(0, 1, num_points)
    for i in range(n + 1):
        binomial_coeff = np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))
        bezier_curve += binomial_coeff * (t ** i)[:, None] * ((1 - t) ** (n - i))[:, None] * points[i]
    return bezier_curve


# def plot_fitting_methods(data, dbscan_points, poly_points, b_spline_points, bezier_points):
#     plt.figure(figsize=(12, 6))
#     ax = plt.axes(projection="3d")

#     # Plot original points
#     ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='gray', s=1, alpha=0.5, label="Original Points")

#     # Plot DBSCAN interpolated points
#     dbscan_points = np.array(dbscan_points)
#     # ax.plot(dbscan_points[:, 0], dbscan_points[:, 1], dbscan_points[:, 2], '-r', label="DBSCAN Interpolation")

#     # # Plot Polynomial points
#     # ax.plot(poly_points[:, 0], poly_points[:, 1], poly_points[:, 2], '-m', label="Polynomial Fit")

#     # # Plot B-spline points
#     # ax.plot(b_spline_points[:, 0], b_spline_points[:, 1], b_spline_points[:, 2], '-g', label="B-spline Fit")

#     # # Plot Bezier points
#     # ax.plot(bezier_points[:, 0], bezier_points[:, 1], bezier_points[:, 2], '-b', label="Bezier Fit")

#     ax.set_xlabel("X",labelpad=20)
#     ax.set_ylabel("Y",labelpad=50)
#     ax.set_zlabel("Z",labelpad=20)
#     # ax.set_title("Fitting Methods Comparison")
#     ax.set_box_aspect([1, 5, 1])
#     # ax.legend()
#     # plt.show()
#     fig = go.Figure()

#     # Add traces for each method
#     fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2],
#                             mode='markers', marker=dict(size=2, color='gray'),
#                             name="Original Points"))

#     fig.add_trace(go.Scatter3d(x=dbscan_points[:, 0], y=dbscan_points[:, 1], z=dbscan_points[:, 2],
#                             mode='lines', line=dict(color='red'), name="DBSCAN"))

#     fig.add_trace(go.Scatter3d(x=poly_points[:, 0], y=poly_points[:, 1], z=poly_points[:, 2],
#                             mode='lines', line=dict(color='magenta'), name="Polynomial Fit"))

#     fig.add_trace(go.Scatter3d(x=b_spline_points[:, 0], y=b_spline_points[:, 1], z=b_spline_points[:, 2],
#                             mode='lines', line=dict(color='green'), name="B-spline Fit"))

#     fig.add_trace(go.Scatter3d(x=bezier_points[:, 0], y=bezier_points[:, 1], z=bezier_points[:, 2],
#                             mode='lines', line=dict(color='blue'), name="Bezier Fit"))

#     fig.update_layout(
#         scene=dict(
#             aspectratio=dict(x=1, y=5, z=1),  # Y-axis 5 times longer than X and Z
#             xaxis_title="X",
#             yaxis_title="Y",
#             zaxis_title="Z"
#         ),
#         title="Fitting Methods Comparison",
#         )   
    
#     # Show the interactive plot
#     # fig.show()

class RopeSegmenter:
    def __init__(self, video_dir= "notebooks/images", depth_dir= "notebooks/depth_images"):

        # Initialize variables
        self.udp_host_1 = '10.157.174.101'
        self.udp_host_2 = '10.157.175.179'
        self.udp_port = 5060

        self.device = self.select_device()
        self.predictor = self.load_predictor()
        
        self.video_dir = video_dir
        self.depth_dir = depth_dir
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)
        if not os.path.exists(self.depth_dir):
            os.makedirs(self.depth_dir)

        self.rotation_matrix_cam_to_base = np.array([[-0.00188135,  0.72490313 ,-0.68884825  ,1.06138857],
                                                    [ 0.99894452 ,-0.03025217 ,-0.03456386  ,0.02760325],
                                                    [-0.0458946 , -0.68818621 ,-0.72408109 , 0.65305965],
                                                    [ 0.     ,     0.      ,    0.    ,      1.        ]])
        
        self.x_min, self.x_max = -2, 2
        self.y_min, self.y_max = -2, 2
        self.z_min, self.z_max = 0.2, 2
        # self.y_threshold = 0.1

        self.depth_map_filter_size = 5
        self.poly_order = 5
        
        self.intrinsics = None

        self.frame_count = 0
        self.max_init_frame = 2
        self.frame0_saved = False
        self.reset_frames = False
        self.frames_collected = False
        self.initial_points = None
        self.tracked_grasp_point = None

        self.bridge = CvBridge()

        if not rospy.core.is_initialized():
            rospy.init_node('rope_segmenter_node', anonymous=True)
            rospy.loginfo("Initialized ROS node 'rope_segmenter_node'")
        else:
            rospy.loginfo("ROS node already initialized")

        self.rope_marker_pub = rospy.Publisher('rope_marker', Marker, queue_size=100)
        self.intersections_marker_pub = rospy.Publisher('intersections_marker', Marker, queue_size=100)
        self.grasp_point_pub = rospy.Publisher('grasp_point_marker', Marker, queue_size=100)
        self.wo_intersection_point_pub = rospy.Publisher('wo_intersection_point_pub', Marker, queue_size=100)

        self.init_grasp_flag = False

        self.frame1_points = np.array([]).reshape(0, 2) 
        self.frame2_points = np.array([]).reshape(0, 2)

        self.initial_translation_direction = None 
        
    def select_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\n mps"
            )
        return device

    def load_predictor(self):
        from sam2.build_sam import build_sam2_video_predictor
        # print("Current working directory:", os.getcwd())
        sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        return predictor

    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    def intrinsics_callback(self, msg):
        self.intrinsics = (msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4])
        fx, fy, cx, cy, depth_scale = self.intrinsics
        rospy.loginfo(f"Received intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_scale={depth_scale}")

    def get_intrinsics_from_ros(self):
        rospy.Subscriber('realsense/camera_intrinsics', Float64MultiArray, self.intrinsics_callback)
        rate = rospy.Rate(10)  # Check at 10 Hz
        while self.intrinsics is None and not rospy.is_shutdown():
            rospy.loginfo("Waiting for intrinsics...")
            rate.sleep()
        return self.intrinsics

    def save_frame(self, color_msg, depth_msg, filename_prefix):
        try:
            # Convert ROS Image messages to OpenCV images
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            # Define filenames
            color_filename = os.path.join(self.video_dir, f"{filename_prefix}.jpg")
            depth_filename = os.path.join(self.depth_dir, f"{filename_prefix}_depth.npy")

            # Save color image using PIL for consistent RGB format
            PILImage.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)).save(color_filename)
            rospy.loginfo(f"Saved color image: {color_filename}")

            # Save depth image as a NumPy array
            np.save(depth_filename, depth_image)
            rospy.loginfo(f"Saved depth image: {depth_filename}")

        except Exception as e:
            rospy.logerr(f"Error saving frame {filename_prefix}: {e}")

    def image_callback(self, color_msg, depth_msg):
        try:
            rospy.loginfo("Image callback triggered.")

            if not self.frame0_saved:
                rospy.loginfo("Saving frame0")
                self.save_frame(color_msg, depth_msg, "0")
                self.frame0_saved = True
                return

            if self.reset_frames:
                if self.frame_count >= self.max_init_frame:
                    # Completed updating 20 frames; reset the flag
                    rospy.loginfo(f"Updated 1 to {self.max_init_frame} frames. Stopping updates.")
                    self.frames_collected = True
                    self.reset_frames = False
                    return

            rospy.loginfo(f"Frame {self.frame_count}: Image callback triggered")
            self.save_frame(color_msg, depth_msg, str(self.frame_count))
            self.frame_count += 1

        except Exception as e:
            rospy.logerr(f"Error processing images: {e}")

    def collect_frames(self, color_topic="/realsense/color_image", depth_topic="/realsense/depth_image"):
        # Delete previous frames except frame 0
        for filename in os.listdir(self.video_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                basename = os.path.splitext(filename)[0]
                if basename.isdigit() and int(basename) != 0:
                    os.remove(os.path.join(self.video_dir, filename))
        for filename in os.listdir(self.depth_dir):
            if filename.endswith(".npy") or filename.endswith(".png"):
                basename = os.path.splitext(filename)[0].replace("_depth", "")
                if basename.isdigit() and int(basename) != 0:
                    os.remove(os.path.join(self.depth_dir, filename))

        self.frame_count = 1  # Start from frame 1
        self.reset_frames = True
        self.frames_collected = False

        color_sub = message_filters.Subscriber(color_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)

        # Synchronize the topics
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.image_callback)

        rospy.loginfo("Subscribers set up. Starting to collect frames...")
        while not self.frames_collected and not rospy.is_shutdown():
            rospy.sleep(0.1)

        color_sub.unregister()
        depth_sub.unregister()

        rospy.loginfo("Frame collection complete.")

    def generate_2d_rope_model(self, out_mask_logits, method='mean'):
        # Extract 2D Pixel Positions of the Mask
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()  # Convert mask to numpy
        pixel_positions = np.column_stack(np.where(mask))  # Get (c, y, x) positions of mask pixels
        
        # Extract x and y coordinates
        x_coords = pixel_positions[:, 2]
        y_coords = pixel_positions[:, 1]
        
        # Use pandas for efficient grouping and median calculation
        df = pd.DataFrame({'x': x_coords, 'y': y_coords})
        if method == 'mean':
            df_2d= df.groupby('x')['y'].median().reset_index()
        elif method == 'median':
            df_2d = df.groupby('x')['y'].median().reset_index()
        else:
            print("Please give a method to generate 2d DLO modle.")
        
        # Convert the DataFrame to a list of tuples and sort by x
        pixel_positions_sorted_asc = df_2d.values.tolist()

        print(f'median_pixel_positions: [{len(pixel_positions_sorted_asc)}]') 

        # self.plot_2d_points(pixel_positions_sorted_asc) 

        return pixel_positions_sorted_asc

    def plot_2d_points(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

    
        plt.figure(figsize=(10, 5))
        plt.plot(x_coords, y_coords, linestyle='-', marker='o', color='blue', label='2D Rope Model')

        for i in range(len(x_coords) - 1):
            if abs(x_coords[i] - x_coords[i+1]) > 1: 
                plt.plot(x_coords[i+1:], y_coords[i+1:], linestyle='-', marker='o', color='blue')

        plt.xlabel('X Position',labelpad=20)
        plt.ylabel('Y Position',labelpad=20)
        plt.title('2D DLO Model')
        plt.legend()
        plt.gca().invert_yaxis() 
        plt.show()

    def plot_3d_points(self, points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = points[:, 0]
        ys = points[:, 1]
        zs = points[:, 2]

        ax.scatter(xs, ys, zs, s=1, c=zs, cmap='viridis') 
        ax.set_xlabel('X (Base Frame)',labelpad=20)
        ax.set_ylabel('Y (Base Frame)',labelpad=50)
        ax.set_zlabel('Z (Base Frame)',labelpad=20)
        ax.set_title('3D Rope Model in Base Frame')

        ax.set_box_aspect([1, 5, 1])

        plt.show()

    def get_rope_3d_positions_base(self, averaged_pixel_positions_sorted_asc, depth_map):

        fx, fy, cx, cy, depth_scale = self.intrinsics



        # Extract u and v coordinates as arrays
        u_array = np.array([p[0] for p in averaged_pixel_positions_sorted_asc])
        v_array = np.array([p[1] for p in averaged_pixel_positions_sorted_asc])

        # Ensure coordinates are integers for indexing
        u_int = u_array.astype(int)
        v_int = v_array.astype(int)

        # Get depth values and apply depth scale
        depth_values = depth_map[v_int, u_int] * depth_scale

        # Filter out invalid depth values
        valid_mask = depth_values > 0
        u_valid = u_array[valid_mask]
        v_valid = v_array[valid_mask]
        depth_values_valid = depth_values[valid_mask]

        # Compute X, Y, Z coordinates in a vectorized manner
        X = (u_valid - cx) * depth_values_valid / fx
        Y = (v_valid - cy) * depth_values_valid / fy
        Z = depth_values_valid

        # Construct homogeneous coordinates
        points_cam = np.column_stack((X, Y, Z, np.ones_like(X)))

        # Transform points to the base frame
        points_base = points_cam @ self.rotation_matrix_cam_to_base.T

        # Extract the 3D positions in the base frame
        rope_3d_positions_base = points_base[:, :3]

        # self.plot_3d_points(rope_3d_positions_base)

        print("3D positions of the rope in the base frame:")
        print(len(rope_3d_positions_base))

        return rope_3d_positions_base

    def calculate_distance(self, point1, point2):

        return np.sqrt((point2[0] - point1[0]) ** 2 + 
                       (point2[1] - point1[1]) ** 2 + 
                       (point2[2] - point1[2]) ** 2)


    # def DBSCAN_filter(self, rope_3d_positions_base):
        
    #     # Set 3D workspace limit range
    #     rope_3d_positions_base = [
    #         point for point in rope_3d_positions_base
    #         if self.x_min <= point[0] <= self.x_max and
    #         self.y_min <= point[1] <= self.y_max and
    #         self.z_min <= point[2] <= self.z_max
    #     ]

    #     print(f"Filtered rope_3d_positions_base (length={len(rope_3d_positions_base)})")

    #     if len(rope_3d_positions_base) == 0:
    #         print("No points found in the specified range!")
    #         return

    #     # Use DBSCAN clustering
    #     data = np.array(rope_3d_positions_base)
    #     # clustering = DBSCAN(eps=0.05, min_samples=45, algorithm="ball_tree").fit(data)
    #     clustering = DBSCAN(eps=0.05, min_samples=45, algorithm="ball_tree").fit(data)
    #     labels = clustering.labels_

    #     # Create clusters excluding noise
    #     unique_labels = set(labels)
    #     clusters = {label: data[labels == label] for label in unique_labels if label != -1}
    #     print(f"Number of clusters (excluding noise): {len(clusters)}")


    #     if not clusters:
    #         print("No clusters found. Skipping DBSCAN filtering.")
    #         # Return empty fig and interpolated_points
    #         interpolated_points = []
    #         return interpolated_points

    #     # Fit and complete each segment
    #     interpolated_points = []
    #     sorted_clusters = sorted(clusters.values(), key=lambda x: np.mean(x[:, 1]))  # Sort based on Y coordinate

    #     if not sorted_clusters:
    #         print("Sorted clusters are empty.")
    #         return

    #     for i in range(len(sorted_clusters) - 1):
    #         cluster1 = sorted_clusters[i]
    #         cluster2 = sorted_clusters[i + 1]

    #         # Add the first cluster segment to the final list
    #         interpolated_points.extend(cluster1.tolist())

    #         # Calculate the gap distance between the two segments
    #         gap_distance = np.linalg.norm(cluster1[-1] - cluster2[0])

    #         if gap_distance > 0.01:  # 
    #             # Use curve_fit to fit a nonlinear curve and generate a smooth transition segment
    #             fit_points = np.vstack([cluster1[50:], cluster2[:50]])  # Use 30 points from each cluster
    #             y = fit_points[:, 1]  # Use Y as input feature
    #             x = fit_points[:, 0]  # Fit X and Z
    #             z = fit_points[:, 2]

    #             # Use curve_fit for nonlinear curve fitting
    #             params_x, _ = curve_fit(lambda y, *p: self.poly_curve(y, *p), y, x, p0=[1] * (self.poly_order + 1))
    #             params_z, _ = curve_fit(lambda y, *p: self.poly_curve(y, *p), y, z, p0=[1] * (self.poly_order + 1))

    #             # Generate intermediate points
    #             num_missing = 100  # Number of interpolation points
    #             y_missing = np.linspace(cluster1[-1][1], cluster2[0][1], num_missing)

    #             # Use fitting parameters to generate X and Z for intermediate points
    #             x_missing = self.poly_curve(y_missing, *params_x)
    #             z_missing = self.poly_curve(y_missing, *params_z)

    #             # Fill in the missing segment
    #             interpolated_segment = np.column_stack((x_missing, y_missing, z_missing))
    #             interpolated_points.extend(interpolated_segment.tolist())

    #     # Add the last cluster segment
    #     if sorted_clusters:
    #         interpolated_points.extend(sorted_clusters[-1].tolist())
        
    #     self.plot_clusters_and_interpolated_points(data, labels, interpolated_points)

    #     return interpolated_points

    def DBSCAN_filter(self, rope_3d_positions_base):
        # Filter the 3D points within the workspace limits
        rope_3d_positions_base = [
            point for point in rope_3d_positions_base
            if self.x_min <= point[0] <= self.x_max and
            self.y_min <= point[1] <= self.y_max and
            self.z_min <= point[2] <= self.z_max
        ]

        if len(rope_3d_positions_base) == 0:
            print("No points found in the specified range!")
            return

        data = np.array(rope_3d_positions_base)
        clustering = DBSCAN(eps=0.05, min_samples=50, algorithm="ball_tree").fit(data)
        
        # clustering = DBSCAN(eps=0.05, min_samples=125, algorithm="ball_tree").fit(data) #ORGINAL DATA
        
        labels = clustering.labels_

        unique_labels = set(labels)
        clusters = {label: data[labels == label] for label in unique_labels if label != -1}

        if not clusters:
            print("No clusters found. Skipping DBSCAN filtering.")
            return []

        # Interpolate using DBSCAN clusters
        interpolated_points = []
        sorted_clusters = sorted(clusters.values(), key=lambda x: np.mean(x[:, 1]))  # Sort by Y-coordinate

        for i in range(len(sorted_clusters) - 1):
            interpolated_points.extend(sorted_clusters[i].tolist())
        if sorted_clusters:
            interpolated_points.extend(sorted_clusters[-1].tolist())

        interpolated_points = np.array(interpolated_points)

        plt.figure(figsize=(10, 8))

    
        for label, points in clusters.items():
            plt.scatter(points[:, 0], points[:, 1], points[:, 2], label=f"Cluster {label}")

        # 画出被认为是噪声的点
        noise_points = data[labels == -1]
        if len(noise_points) > 0:
            plt.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], color='k', marker='x', label="Noise")

        # plt.title("DBSCAN Filtered Rope Points")
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.legend()
        # plt.show()
        
        # Fit using Polynomial
        poly_points = fit_polynomial(interpolated_points, degree=self.poly_order)
        # self.plot_3d_points(poly_points)
        # Fit using B-spline
        b_spline_points = fit_b_spline(interpolated_points)
        # self.plot_3d_points(b_spline_points)
        # Fit using Bezier
        bezier_points = fit_bezier(interpolated_points)
        # self.plot_3d_points(bezier_points)
        # Visualize all methods
        # plot_fitting_methods(data, interpolated_points, poly_points, b_spline_points, bezier_points)
        # self.plot_3d_points(interpolated_points)
        return interpolated_points
    
    # def plot_clusters_and_interpolated_points(self, data, labels, interpolated_points):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     unique_labels = set(labels)
    #     for label in unique_labels:
    #         if label == -1:
    #             # noise
    #             color = 'grey'
    #             size = 1
    #         else:
    #             color = plt.cm.Spectral(float(label) / len(unique_labels))
    #             size = 10
    #         class_member_mask = (labels == label)
    #         xyz = data[class_member_mask]
    #         # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=[color], s=size)

    #     interpolated_points = np.array(interpolated_points)
    #     ax.plot(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2], c='blue', linewidth=2)

    #     # ax.set_xlabel('X')
    #     # ax.set_ylabel('Y')
    #     # ax.set_zlabel('Z')
    #     ax.set_title('DBSCAN Clustering and Interpolated Rope Points')
    #     ax.set_box_aspect([1, 3, 1])
    #     plt.show()

    def plot_clusters_and_interpolated_points(self, data, labels, interpolated_points):
        fig = plt.figure(figsize=(15, 5))

        # Prepare data for B-spline and Bézier curve
        interpolated_points = np.array(interpolated_points)
        y_points = interpolated_points[:, 1]
        x_points = interpolated_points[:, 0]
        z_points = interpolated_points[:, 2]

        # Plot 1: Polynomial fitting
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='grey', s=1, alpha=0.5, label='Data Points')

        ax1.plot(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2], c='blue', linewidth=2, label='Polynomial Fit')
        ax1.set_title('Polynomial Fitting')
        ax1.set_box_aspect([1, 5, 1])

        # Plot 2: B-spline
        ax2 = fig.add_subplot(132, projection='3d')
        tck, u = splprep([x_points, y_points, z_points], s=2)
        u_fine = np.linspace(0, 1, len(interpolated_points))
        x_spline, y_spline, z_spline = splev(u_fine, tck)

        ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c='grey', s=1, alpha=0.5, label='Data Points')
        ax2.plot(x_spline, y_spline, z_spline, c='green', linewidth=2, label='B-spline Fit')
        ax2.set_title('B-spline Fitting')
        ax2.set_box_aspect([1, 5, 1])

        # # Plot 3: Bézier curve
        # ax3 = fig.add_subplot(133, projection='3d')
        # nodes = np.asfortranarray([x_points, y_points, z_points])
        # curve = bezier.Curve(nodes, degree=min(len(x_points) - 1, 3))
        # eval_points = curve.evaluate_multi(np.linspace(0, 1, len(interpolated_points)))

        # ax3.scatter(data[:, 0], data[:, 1], data[:, 2], c='grey', s=1, alpha=0.5, label='Data Points')
        # ax3.plot(eval_points[0, :], eval_points[1, :], eval_points[2, :], c='red', linewidth=2, label='Bézier Fit')
        # ax3.set_title('Bézier Curve Fitting')
        # ax3.set_box_aspect([1, 5, 1])

        # Display all plots
        plt.tight_layout()
        # plt.show()

    def poly_curve(self, y, *params):
        """
        Defines a polynomial model function based on the polynomial order.
        Args:
            y: Input feature (Y-axis values)
            params: Fit parameters (a, b, c, ...)
        Returns:
            Predicted X or Z axis values.
        """
        return sum(params[i] * y ** i for i in range(len(params)))


    # def select_grasp_point(self, rope_3d_positions):
    #     """
    #     Select a grasp point from the fitted 3D rope model, keeping the y-axis around 0.
    #     Args:
    #         rope_3d_positions_base: List of 3D points [(x, y, z), ...] representing the rope.
    #     Returns:
    #         grasp_point: Tuple representing the selected 3D grasp point (x, y, z).
    #     """

    #     if len(rope_3d_positions) == 0:
    #         raise ValueError("No rope points available to select a grasp point!")

    #     # Filter points within a specific region of interest (including y close to 0)
    #     rope_3d_positions_base_in_space = [
    #         point for point in rope_3d_positions
    #         if self.x_min <= point[0] <= self.x_max and
    #         self.y_min <= point[1] <= self.y_max and
    #         self.z_min <= point[2] <= self.z_max and
    #         abs(point[1]) <= self.y_threshold  # y should be around 0 (within a threshold)
    #     ]

    #     if len(rope_3d_positions_base_in_space) == 0:
    #         raise ValueError("No rope points available within the specified region or y-axis constraint!")

    #     # Calculate the middle index from the filtered points
    #     mid_index = len(rope_3d_positions_base_in_space) // 2

    #     # Select the point in the middle of the sorted positions
    #     grasp_point = rope_3d_positions_base_in_space[mid_index]
    #     # delta = [0, 0, 0.01]
    #     # for i in range(3):

    #     #     grasp_point[i] += delta[i]

    #     print(f"Selected Grasp Point (y ~ 0): {[]}")
    #     return grasp_point

    


    def send_grasp_point_via_udp(self, grasp_point):
        """
        Send the selected grasp point to the robot through UDP.
        Args:
            grasp_point: Tuple of (x, y, z) representing the selected grasp point.
        """
        # Create a UDP socket
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        x = float(grasp_point[0])
        y = float(grasp_point[1])
        z = float(grasp_point[2])


        # Pack the data as a formatted string to send (using JSON format for consistency)
        message = json.dumps({
                    "data": {
                        "init_grasp_position": {
                            "x": x,
                            "y": y,
                            "z": z
                        }
                    }
                })
        # Define robot IP and port (use your robot's IP here)
        robot_ip = self.udp_host_1  # IP of your robot (self.udp_host_1 or self.udp_host_2)
        robot_port = self.udp_port

        # Send the grasp point via UDP
        try:
            udp_socket.sendto(message.encode('utf-8'), (robot_ip, robot_port))
            print(f"Grasp point {message} sent to {robot_ip}:{robot_port}")
        except Exception as e:
            print(f"Failed to send grasp point: {e}")
        finally:
            udp_socket.close()

    def depth_data_processing(self, depth_map):
        n = self.depth_map_filter_size

        # Identify missing or invalid depth data (0 or NaN)
        missing_mask = (depth_map == 0) | np.isnan(depth_map)

        # Apply a median filter to the entire depth map
        filtered_depth_map = median_filter(depth_map, size=n)

        # Replace only the missing values with the filtered data
        depth_map[missing_mask] = filtered_depth_map[missing_mask]

        return depth_map


    # def publish_points(self):

    #     receiver = UDPReceiver(self.udp_host_2, self.udp_port)
        
    #     if not rospy.core.is_initialized():
    #         rospy.init_node('intersection_publisher')

    #     # Create publishers for the two points
    #     pub1 = rospy.Publisher('/point_1', PointStamped, queue_size=100)
    #     pub2 = rospy.Publisher('/point_2', PointStamped, queue_size=100)

    #     rate = rospy.Rate(10)  # Publishing rate at 10 Hz (adjust as needed)

    #     try:
    #         while not rospy.is_shutdown():
    #             # Receive new intersection points and tcp data
    #             intersection_w_1, intersection_w_2, tcp = receiver.receive_data()

    #             # Create two PointStamped messages
    #             point_1 = PointStamped()
    #             point_1.header = Header(stamp=rospy.Time.now(), frame_id="map")
    #             point_1.point.x = intersection_w_1[0]
    #             point_1.point.y = intersection_w_1[1]
    #             point_1.point.z = intersection_w_1[2]

    #             point_2 = PointStamped()
    #             point_2.header = Header(stamp=rospy.Time.now(), frame_id="map")
    #             point_2.point.x = intersection_w_2[0]
    #             point_2.point.y = intersection_w_2[1]
    #             point_2.point.z = intersection_w_2[2]

    #             # Publish both points
    #             pub1.publish(point_1)
    #             pub2.publish(point_2)

    #             rate.sleep()  # Sleep to maintain loop rate
    #     except rospy.ROSInterruptException:
    #         pass
    #     finally:
    #         receiver.close_socket()  # Sleep to maintain loop rate

    def publish_intersection_points(self, intersection_1, intersection_2, tcp):
        """
        Publishes the intersection points and TCP as markers.
        :param intersection_1: 3D point [x, y, z] for intersection 1
        :param intersection_2: 3D point [x, y, z] for intersection 2
        :param tcp: 3D point [x, y, z] for TCP
        """
        marker = Marker()
        marker.header.frame_id = "map"  # Replace with your coordinate frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "intersection_points"
        marker.id = 1
        marker.type = Marker.POINTS  # Use POINTS type for individual points
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # Point width
        marker.scale.y = 0.01  # Point height (since it's a sphere-like marker)
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration()  # Keep the marker until it's updated

        # Create and add the points for intersection 1, intersection 2, and tcp
        points = [intersection_1, intersection_2, tcp]

        for point in points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)

        # Publish the marker
        self.intersections_marker_pub.publish(marker)
        rospy.loginfo(f"Published intersection points and TCP.")



    def calculate_translation(self, intersection_1, intersection_2, closest_point_1, closest_point_2):
        """
        Calculate the translation vector needed to align the rope with the intersection points.
        :param intersection_1: First intersection point [x, y, z].
        :param intersection_2: Second intersection point [x, y, z].
        :param closest_point_1: Closest point on the rope model to intersection_1.
        :param closest_point_2: Closest point on the rope model to intersection_2.
        :return: Translation vector [x, y, z].
        """
        # Calculate midpoints of the intersections and closest rope points
        midpoint_intersections = (np.array(intersection_1) + np.array(intersection_2)) / 2
        midpoint_rope = (np.array(closest_point_1) + np.array(closest_point_2)) / 2

        # Compute the translation vector
        translation_vector = midpoint_intersections - midpoint_rope
        return translation_vector
    
    def find_closest_points_sam2(self, rope_points, intersection_1, intersection_2, initial_grasp_point):
        """
        Find the closest points on the SAM2 mask rope model to the given intersection points,
        with corrections using the grasp point and intersection points as ground truth.
        :param rope_points: List of 3D points representing the rope.
        :param intersection_1: First intersection point as [x, y, z].
        :param intersection_2: Second intersection point as [x, y, z].
        :param initial_grasp_point: Ground truth grasp point as [x, y, z].
        :return: Corrected closest points on the rope to intersection_1 and intersection_2.
        """
        rope_points_np = np.array(rope_points)

        # Use NearestNeighbors to find the closest points
        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(rope_points_np)

        # Find the closest point to intersection_1 and calculate offset
        _, index_1 = neighbors.kneighbors([intersection_1])
        closest_point_1 = rope_points_np[index_1[0][0]]
        offset_1 = np.array(intersection_1) - closest_point_1

        # Find the closest point to intersection_2 and calculate offset
        _, index_2 = neighbors.kneighbors([intersection_2])
        closest_point_2 = rope_points_np[index_2[0][0]]
        offset_2 = np.array(intersection_2) - closest_point_2

        # Find the closest point to the initial grasp point and calculate offset
        _, index_grasp = neighbors.kneighbors([initial_grasp_point])
        closest_grasp_point = rope_points_np[index_grasp[0][0]]
        grasp_offset = np.array(initial_grasp_point) - closest_grasp_point

        # Apply average or weighted correction based on ground truth points
        corrected_closest_point_1 = closest_point_1 + (offset_1 + grasp_offset) / 2
        corrected_closest_point_2 = closest_point_2 + (offset_2 + grasp_offset) / 2

        return corrected_closest_point_1, corrected_closest_point_2

    def apply_translation_to_rope(self, rope_points, translation_vector):
        """
        Apply the translation to the entire rope model.
        :param rope_points: List of 3D points representing the rope.
        :param translation_vector: Vector [x, y, z] to translate the rope.
        :return: Translated rope points.
        """
        if rope_points is None:
            print("Error: rope_points is None.")
            return None
    
        rope_points_np = np.array(rope_points)

        # Perform vectorized translation
        translated_rope = rope_points_np + translation_vector

        return translated_rope
    
    def publish_grasp_point(self, grasp_point):
        """
        Publish a marker for the initial grasp point to RViz.
        :param grasp_point: 3D point [x, y, z] representing the initial grasp point.
        """
        marker = Marker()
        marker.header.frame_id = "map"  # Replace with your coordinate frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "grasp_point"
        marker.id = 3 
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set the pose of the marker (the position of the grasp point)
        marker.pose.position.x = grasp_point[0]
        marker.pose.position.y = grasp_point[1]
        marker.pose.position.z = grasp_point[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Customize the size and color of the marker
        marker.scale.x = 0.05  # Sphere radius in meters
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green color
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        # Publish the marker
        self.grasp_point_pub.publish(marker)

    def publish_smoothed_rope_marker(self, rope_points, color):
        smoothed_rope = self.smooth_rope_with_spline(rope_points, smooth_factor=0.05, num_points=30)

        # ROS Marker definition
        marker = Marker()
        marker.header.frame_id = "map"  
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rope_model"
        marker.id = 0
        
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01  
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        
        # Make the line darker if color is green
        if color == [0, 1, 0]:  
            marker.color.a = 0.5  # Darker by reducing alpha
        else:
            marker.color.a = 1.0  # Full opacity for other colors

        marker.lifetime = rospy.Duration()

        # Add points with a dotted effect
        for i, point in enumerate(smoothed_rope):
            if i % 2 == 0:  # Skip every other point to create dotted effect
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = point[2]
                marker.points.append(p)

        # Publish the smoothed rope model
        if color == [0, 1, 0]:
            self.wo_intersection_point_pub.publish(marker)  # Green, darker, dotted
        else:
            self.rope_marker_pub.publish(marker)  # Default solid line

    def smooth_rope_with_spline(self, rope_3d_positions_base, smooth_factor=0.05, num_points=25):

        x = [p[0] for p in rope_3d_positions_base]
        y = [p[1] for p in rope_3d_positions_base]
        z = [p[2] for p in rope_3d_positions_base]
        
        # print('smooth_xyz',x,y,z)
        
        tck, u = splprep([x, y, z], s=smooth_factor)
        u_fine = np.linspace(0, 1, num_points)  
        x_smooth, y_smooth, z_smooth = splev(u_fine, tck)

        smoothed_rope = list(zip(x_smooth, y_smooth, z_smooth))
        return smoothed_rope

    def select_initial_grasp_point(self, image_path):
        """
        Display the image and allow the user to select the initial grasp point.
        :param image_path: Path to the image file.
        :return: Selected grasp point in pixel coordinates [x, y].
        """
        image = PILImage.open(image_path)
        plt.figure(figsize=(9, 6))
        plt.title("Select the initial grasp point")
        plt.imshow(image)

        # Use ginput to let the user select one point
        grasp_point_2d = plt.ginput(1)  # Allows the user to choose 1 point
        grasp_point_2d=[(640,360)]
        plt.close()

        # Convert the selected point to numpy array
        grasp_point_2d = np.array(grasp_point_2d[0], dtype=np.float32)
        print(f"User selected grasp point: {grasp_point_2d}")

        return grasp_point_2d
    
    def resize_depth_map(self, depth_map, target_size=(1280, 720)):
        """
        Resize the depth map to match the target (RGB image) size.
        :param depth_map: Original depth map.
        :param target_size: Target size (width, height) as a tuple.
        :return: Resized depth map.
        """
        return cv2.resize(depth_map, target_size, interpolation=cv2.INTER_NEAREST)

    def project_2d_to_3d(self, point_2d, depth_map):
        """
        Project a 2D point into 3D space using the resized depth map and updated intrinsics.
        :param point_2d: 2D point [x, y] in pixel coordinates.
        :param depth_map: Original depth map.
        :param intrinsics: Camera intrinsics [fx, fy, cx, cy, depth_scale].
        :return: 3D point [x, y, z] in camera coordinate system.
        """
        # Resize depth map to RGB size
        depth_map_resized = self.resize_depth_map(depth_map, target_size=(1280, 720))
        
        # Adjust intrinsics based on the scaling factors
        fx, fy, cx, cy, depth_scale = self.intrinsics
        
        x, y = point_2d
        x = int(round(x))
        y = int(round(y))

        # Get depth value at the pixel
        if 0 <= y < depth_map_resized.shape[0] and 0 <= x < depth_map_resized.shape[1]:
            depth = depth_map_resized[y, x] * depth_scale
        else:
            print("Selected point is outside the depth map bounds.")
            return None

        if depth == 0:
            print("Depth at the selected point is zero. Cannot compute 3D point.")
            return None

        # Compute the 3D coordinates
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        grasp_point_3d_cam = np.array([X, Y, Z], dtype=np.float32)

        # Convert to base frame
        grasp_point_homogeneous = np.append(grasp_point_3d_cam, 1)
        grasp_point_base = grasp_point_homogeneous @ self.rotation_matrix_cam_to_base.T
        grasp_point_base = grasp_point_base[:3]

        return grasp_point_base
    

    def correct_rope_model(self, rope_points, intersection_1, intersection_2):
        """
        Correct the rope model to pass accurately through the given intersection points using translation only.
        :param rope_points: List of 3D points representing the initial rope model.
        :param intersection_1: First intersection point as [x, y, z] (ground truth).
        :param intersection_2: Second intersection point as [x, y, z] (ground truth).
        :return: Corrected rope model points.
        """
        rope_points_np = np.array(rope_points)

        # Step 1: Find the closest points on the rope model to intersection_1 and intersection_2
        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(rope_points_np)

        _, index_1 = neighbors.kneighbors([intersection_1])
        closest_point_1 = rope_points_np[index_1[0][0]]

        _, index_2 = neighbors.kneighbors([intersection_2])
        closest_point_2 = rope_points_np[index_2[0][0]]

        # Step 2: Calculate the midpoint translation vector based on the two intersections
        midpoint_true = (np.array(intersection_1) + np.array(intersection_2)) / 2
        midpoint_model = (closest_point_1 + closest_point_2) / 2

        # Calculate the translation to align the model's midpoint to the ground truth midpoint
        translation = midpoint_true - midpoint_model

        # # Step 3: Ensure the translation follows the initial direction
        # if self.initial_translation_direction is None:
        #     # Store the direction of the first correction
        #     self.initial_translation_direction = translation / np.linalg.norm(translation)
        # else:
        #     # Check if the current translation aligns with the initial direction
        #     if np.dot(translation, self.initial_translation_direction) < 0:
        #         # Reverse translation if it goes in the opposite direction
        #         translation = -translation

        translated_rope = rope_points_np + translation
        return translated_rope

    def show_depth_map(self, depth_map):
        depth_map_normalized_original = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_uint8_original = np.uint8(depth_map_normalized_original)
        depth_colormap_original = cv2.applyColorMap(depth_map_uint8_original, cv2.COLORMAP_JET)

        depth_map_normalized_filtered = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_uint8_filtered = np.uint8(depth_map_normalized_filtered)
        depth_colormap_filtered = cv2.applyColorMap(depth_map_uint8_filtered, cv2.COLORMAP_JET)

        combined_depth_map = np.hstack((depth_colormap_original, depth_colormap_filtered))

        scale_percent = 50 
        width = int(combined_depth_map.shape[1] * scale_percent / 100)
        height = int(combined_depth_map.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_combined_depth_map = cv2.resize(combined_depth_map, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Original and Filtered Depth Map", resized_combined_depth_map)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()

    
    def replace_and_fit_spline(self, rope_points, intersection_1, intersection_2):
        rope_points = np.array(rope_points)
        neighbors = NearestNeighbors(n_neighbors=1)
        neighbors.fit(rope_points)
        _, idx1 = neighbors.kneighbors([intersection_1])
        _, idx2 = neighbors.kneighbors([intersection_2])
        idx1, idx2 = sorted([idx1[0][0], idx2[0][0]])

        updated_points = np.vstack([
            rope_points[:idx1],
            intersection_1,
            intersection_2,
            rope_points[idx2+1:]
        ])

        x, y, z = updated_points.T
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
        u = np.hstack(([0], np.cumsum(distances)))
        u /= u[-1]

        tck, _ = splprep([x, y, z], u=u, s=0.1)
        u_fine = np.linspace(0, 1, 300)
        x_spline, y_spline, z_spline = splev(u_fine, tck)

        corrected_rope = np.vstack([x_spline, y_spline, z_spline]).T

        return corrected_rope

    
    def read_newest_data(self, file_path):
        """
        Read the newest line from a text file.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The newest line from the file, or None if the file is empty or does not exist.
        """
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()
                if lines:
                    
                    line = lines[-1].strip() 
                    data_list = list(map(float, line.split(",")))
                    intersection1=data_list[:3]
                    intersection2=data_list[3:6]
                    tcp=data_list[6:9]

                    return intersection1,intersection2,tcp
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading file: {e}")
        return None

    def save_to_file(self, intersection_w_1, intersection_w_2, tcp, translation_vector, filename="rope_data_y.json"):
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert ndarray to list
            elif isinstance(obj, dict):  # Recursively handle dictionaries
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):  # Recursively handle lists
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj  # Return the object if no conversion is needed
        distance = self.calculate_translation_distance(translation_vector)
        # Create a dictionary to store the data
        new_data = {
            "intersection_points": {
                "intersection_w_1": convert_to_serializable(intersection_w_1),
                "intersection_w_2": convert_to_serializable(intersection_w_2),
                "tcp": convert_to_serializable(tcp)
            },
            "translation_distance": convert_to_serializable(distance),
            
        }
        
        if os.path.exists(filename):
            # If file exists, read its content
            with open(filename, "r") as f:
                try:
                    data = json.load(f)  # Load existing data
                except json.JSONDecodeError:
                    data = []  # Start with an empty list if file is corrupted or empty
        else:
            data = []  # If file does not exist, start with an empty list

        # Append the current data to the list
        data.append(new_data)

        # Write the updated data back to the file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


    def calculate_translation_distance(self, translation_vector):
        # Ensure the input is a numpy array
        translation_vector = np.array(translation_vector)
        
        # Calculate the total distance (magnitude)
        distance = np.linalg.norm(translation_vector)
        
        return distance,

    def main(self):

        root = ET.Element("RunTimes")
        receiver = UDPReceiver(self.udp_host_2, self.udp_port, '/home/tp2/Documents/sam2/intersections.txt')
        
        # Get intrinsics
        self.get_intrinsics_from_ros()
        
        fx, fy, cx, cy, depth_scale = self.intrinsics
        scale_x = 1280 / 848  
        scale_y = 720 / 480   

        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        updated_intrinsics = (fx, fy, cx, cy, depth_scale)

        self.intrinsics = updated_intrinsics

        rospy.loginfo(f"Updated intrinsics for resized depth map: fx={fx}, fy={fy}, cx={cx}, cy={cy}, depth_scale={depth_scale}")

        video_segments = {}
        # Initialize frame counter and the initial frame setup
        frame_idx = 0

        # Collect frames continuously until shut down
        while not rospy.is_shutdown():
            #==================================================start time================================================================
            start_time = time.time()

            rospy.loginfo("Starting to collect frames...")
            self.collect_frames()

            while not self.frames_collected and not rospy.is_shutdown():
                rospy.sleep(0.1)

            # Read frame names
            frame_names = [
                p for p in os.listdir(self.video_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"] and os.path.splitext(p)[0].isdigit()
            ]
            
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            
            rospy.loginfo("Processing collected frames...")


            # Display the initial frame for user to interact
            if frame_idx == 0:
                if frame_names:
                    frame_path = os.path.join(self.video_dir, frame_names[frame_idx])
                    image = PILImage.open(frame_path)

                    if self.initial_points is None:
                    # Display the frame using matplotlib
                        plt.figure(figsize=(18, 12))
                        plt.title(f"Select two points in frame {frame_idx}")
                        plt.imshow(image)

                        # Use ginput to let the user select two points

                        self.initial_points = plt.ginput(1)  # Allows the user to choose 2 points
                        plt.close()  

                        # Convert the selected points to numpy array
                        self.initial_points = np.array(self.initial_points, dtype=np.float32)
                        print(f"User selected points: {self.initial_points}")
                else:
                    return print('No frame found in the directory.')

                # Initialize predictor state for the first frame
                inference_state = self.predictor.init_state(video_path=self.video_dir)
                self.predictor.reset_state(inference_state)

                # Using the selected points in the initial interaction
                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
                # Labels for the points, '1' for positive clicks
                labels = np.array([1], np.int32)

                # Add the new user-selected points
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=self.initial_points,
                    labels=labels,
                )

                # Show the results on the current (interacted) frame
                plt.figure(figsize=(9, 6))
                plt.title(f"Frame {ann_frame_idx}")
                plt.imshow(PILImage.open(os.path.join(self.video_dir, frame_names[ann_frame_idx])))
                self.show_points(self.initial_points, labels, plt.gca())
                self.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
                # plt.show()

            if frame_names:

                inference_state = self.predictor.init_state(video_path=self.video_dir)
                self.predictor.reset_state(inference_state)

                # Using the selected points in the initial interaction
                ann_frame_idx = 0  # the frame index we interact with
                ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

                # Labels for the points, '1' for positive clicks
                labels = np.array([1], np.int32)

                # Add the new user-selected points
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=self.initial_points,
                    labels=labels,
                )


                # Process only the last frame
                last_frame_name = frame_names[-1]  # Get the last frame name
                frame_path = os.path.join(self.video_dir, last_frame_name)

                rospy.loginfo(f"Processing the last frame: {last_frame_name}")

                # Update the state and propagate using the last frame
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(0, len(frame_names), 1):
                    plt.figure(figsize=(6, 4))
                    plt.title(f"frame {out_frame_idx}")
                    plt.imshow(PILImage.open(os.path.join(self.video_dir, frame_names[out_frame_idx])))
                # Generate 2D rope model using the mask logits
                averaged_pixel_positions_sorted_asc = self.generate_2d_rope_model(out_mask_logits,'mean')

                # Load the corresponding depth map for the last frame
                ann_frame_idx = int(os.path.splitext(last_frame_name)[0])
                depth_filename = os.path.join(self.depth_dir, f"{ann_frame_idx}_depth.npy")

                depth_map = np.load(depth_filename, allow_pickle=True)
                depth_map_resized = self.resize_depth_map(depth_map, target_size=(1280, 720))
                depth_map_filtered = self.depth_data_processing(depth_map_resized)
                
                # Show the depth map filtered
                # self.show_depth_map(depth_map_resized)
                # cv2.waitKey(1)  # Adjust this if you need it to auto-close or wait for a keypress


                # Update 3D rope model and select grasp point
                rope_3d_positions_base = self.get_rope_3d_positions_base(averaged_pixel_positions_sorted_asc, depth_map_filtered)
                interpolated_points = self.DBSCAN_filter(rope_3d_positions_base)


                # --------------------- Initial Grasp -----------------------
                
                if not self.init_grasp_flag:
                    grasp_point_2d = self.select_initial_grasp_point(frame_path)
                    # initial grasp position based on camera reconstuct 3D point in base frame
                    self.init_grasp_point = self.project_2d_to_3d(grasp_point_2d, depth_map_filtered)
                    self.send_grasp_point_via_udp(self.init_grasp_point)
                    self.init_grasp_flag = True

                # # self.init_grasp_point = [0.629,-0.011,0.54]
                intersection_w_1, intersection_w_2, tcp = self.read_newest_data('/home/tp2/Documents/sam2/intersections.txt')
            
                
                # # # center_of_grasp = (np.array(intersection_w_1) + np.array(intersection_w_2)) / 2
                # # # translation_vector = center_of_grasp - tcp
                # closest_point_1, closest_point_2 = self.find_closest_points_sam2(interpolated_points, intersection_w_1, intersection_w_2, self.init_grasp_point)
                # # corrected_rope_points = self.align_and_correct_rope_model(interpolated_points, intersection_w_1, intersection_w_2)
                corrected_rope_points = self.correct_rope_model(interpolated_points,intersection_w_1, intersection_w_2)
                # # # Calculate the translation vector to correct the rope position
                # translation_vector = self.calculate_translation(intersection_w_1, intersection_w_2, closest_point_1, closest_point_2)
                translation = [0, 0.281 * 2,0]
                # # # Apply the translation to the entire rope model
                # corrected_rope_points = self.apply_translation_to_rope(interpolated_points, translation_vector)

                
                self.publish_intersection_points(intersection_w_1, intersection_w_2, tcp)
                self.publish_smoothed_rope_marker(corrected_rope_points, [1,0,0]) #red
                # self.publish_grasp_point(self.init_grasp_point)
                self.publish_smoothed_rope_marker(interpolated_points, [0,1,0]) #green
                
                #-------------------------------------------------------------
                print('corrected points', corrected_rope_points)
                grasp_robot_2 = corrected_rope_points[int(len(corrected_rope_points)/3)] 
                print('grasp_robot_2', grasp_robot_2)
                #==================================================end time================================================================
                
                # rospy.sleep(0.1)
                end_time = time.time()
                runtime = end_time - start_time
                run_element = ET.SubElement(root, "Run")
                run_element.set("FrameIndex", str(frame_idx))
                runtime_element = ET.SubElement(run_element, "Runtime")
                runtime_element.text = str(runtime)
                print(f"\n\n ----------------------------------Runtime---------------------------------:\n {runtime} seconds\n\n")
                frame_idx += 1
                
                # self.select_grasp_point(corrected_rope_points,translation)
                
          
                    
            tree = ET.ElementTree(root)
            tree.write("runtimes.xml")
            print("All runtimes have been saved to runtimes.xml")  
            frame_idx = 1

        print("main is end")



if __name__ == '__main__':

    torch.cuda.empty_cache()
    segmenter = RopeSegmenter()

    segmenter.main()
import cv2
import os
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import open3d as o3d
import timeit
import sys

# Load JSON file
with open("point_cloud.json", "r") as f:
    data = json.load(f)

# Z IS Y (map from 0 to 720), Y IS X (map from 0 to 1280)
# POINT CLOUD PREPROCESSING
cloud = np.array(data["points"])
cloud = cloud[cloud[:, 2] < 1.0]
cloud_rho = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2 + cloud[:, 2]**2)
cloud_theta = np.arctan(cloud[:, 1]/cloud[:, 0])
cloud_phi = np.arccos(cloud[:, 2]/cloud_rho)

# Point cloud in spherical coordinates, rad = depth.
cloud_sphr = np.array([cloud_rho, cloud_theta, cloud_phi]).T
cloud_sphr = cloud_sphr[cloud_sphr[:, 0] < 1.5]
cloud_sphr = cloud_sphr[cloud_sphr[:, 0] > 0.5]
cloud_sphr = cloud_sphr[cloud_sphr[:, 1] > -math.pi/3]
cloud_sphr = cloud_sphr[cloud_sphr[:, 1] < math.pi/3]
cloud_sphr = cloud_sphr[cloud_sphr[:, 2] > math.pi/4]

np.set_printoptions(suppress=True)

# Given the sample images from ZED are 1280 x 720, most likely using HD resolution.
# [LEFT_CAM_HD]
fx_l=699.41
fy_l=699.365
cx_l=652.8
cy_l=358.133
k1_l=-0.170704
k2_l=0.0249542
p1_l=9.55189e-05
p2_l=-0.000109509
k3_l=-9.96888e-11

# [RIGHT_CAM_HD]
fx_r=697.635
fy_r=697.63
cx_r=671.665
cy_r=354.611
k1_r=-0.171533
k2_r=0.0258402
p1_r=7.86599e-05
p2_r=-0.000136126
k3_r=-2.87251e-10

Baseline=.119905
TY=0.0458908
TZ=-0.467919
CV_2K=0.00697667
RX_2K=0.00239722
RZ_2K=-0.0021326

# Canera matrices.
K_L = np.array([[fx_l, 0, cx_l], [0, fy_l, cy_l], [0, 0, 1]])
K_R = np.array([[fx_r, 0, cx_r], [0, fy_r, cy_r], [0, 0, 1]])
T = np.array([Baseline, TY, TZ])
R, _ = cv2.Rodrigues(np.array([RX_2K, CV_2K, RZ_2K]))
T_x = np.array([[0, -TZ, TY], [TZ, 0, -Baseline], [-TY, Baseline, 0]])
E = T_x @ R
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
dist = np.array([k1_r, k2_r, p1_r, p2_r, k3_r])

RT = np.vstack((np.hstack((R, [[Baseline], [TY], [TZ]])), [0,0,0,1]))

def project_to_2d (cloud, image):
    # graph cut: where vertex = points, edge weights = 1/distance between points (larger distance, lower edge weight)
    z, x, y = cloud[:, 0]*np.sin(cloud[:, 2])*np.cos(cloud[:, 1]), cloud[:, 0]*np.sin(cloud[:, 2])*np.sin(cloud[:, 1]), cloud[:, 0]*np.cos(cloud[:, 2])
    x = np.max(x) - x
    y = np.max(y) - y
    x = x/np.max(x) * 1280
    y = y/np.max(y) * 720/2
    points = np.array([[x],[y],[z]])

    points_2d = np.array(points).T

    # M_intL = np.append(K_L, np.array([[0], [0], [0]]), axis = 1)
    # M_intR = np.append(K_R, np.array([[0], [0], [0]]), axis = 1)

    # M_L = np.array([[M_intL[0][0], M_intL[0][1], M_intL[0][2], M_intL[0][3]],
    #                 [M_intL[1][0], M_intL[1][1], M_intL[1][2], M_intL[1][3]],
    #                 [M_intL[2][0], M_intL[2][1], M_intL[2][2], M_intL[2][3]]])
    # M_R = np.array([[M_intR[0][0], M_intR[0][1], M_intR[0][2], M_intR[0][3]],
    #                 [M_intR[1][0], M_intR[1][1], M_intR[1][2], M_intR[1][3]],
    #                 [M_intR[2][0], M_intR[2][1], M_intR[2][2], M_intR[2][3]]])
    

    # points_4d = np.vstack((points, np.full(shape=points_2d.shape[0], fill_value=1, dtype=np.int)))

    # points_2d = np.dot(M_L, points_4d)

    # x, y, z = points_2d
    # x = x/np.max(x) * 1280
    # y = y/np.max(y) * 720/2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c = 'r', marker = 'o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
    depth = 255 * (points_2d[2] - np.min(points_2d[2])) / (np.max(points_2d[2]) - np.min(points_2d[2])) 
    depth = np.clip(depth, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    for i in range(points_2d.shape[0]):
        u, v = int(x[i]), int(y[i])
        cv2.circle(image, (u, v), 2, (0, 0, 255), -1)  # Draw green dot

    return image

class Node(object):
    def __init__(self, data=None, axis=None, left=None, right=None):
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right

    def make_kdtree (self, points, axis, dim):
        if (points.shape[0] <= 10):
            # left = None, right = None is a Leaf
            return Node(data = points, axis = axis)

        points = points[np.argsort(points[:, axis])]
        median = np.median(points[:, axis])
        left = points[points[:, axis] < median]
        right = points[points[:, axis] >= median]

        return Node(data = points, axis = axis, 
                    left = self.make_kdtree(points = left, axis = (axis + 1) % dim, dim = dim),
                    right = self.make_kdtree(points = right, axis = (axis + 1) % dim, dim = dim))
    
    def search_point(self, point, radius):
        split_axis = np.median(self.data[:, self.axis])
        if (len(self.data[np.linalg.norm(self.data - point[0], axis=1) <= radius]) > 0):
            return point + [self.data[np.linalg.norm(self.data - point[0], axis=1) <= radius]]
        
        if (point[0][self.axis] - radius < split_axis):
            point = self.left.search_point(point, radius)
        elif (point[0][self.axis] + radius >= split_axis):
            point = self.right.search_point(point, radius)

        return point

    def search_tree(self, root, start_point, radius, C):
        stack = [start_point]
        unexplored_set = {tuple(p) for p in Node.unexplored}

        while stack:
            point = stack.pop()
            neighbors = np.vstack(self.search_point([point], radius)[1:])
            neighbors = [tuple(p) for p in neighbors if tuple(p) in unexplored_set]
            
            for neighbor in neighbors:
                if neighbor not in C[-1]:
                    C[-1].append(neighbor)
                    stack.append(np.array(neighbor))  # Add to stack for further exploration
                    unexplored_set.remove(neighbor)

        Node.unexplored = np.array(list(unexplored_set)) if unexplored_set else np.empty((0, 3))
        return Node.unexplored

def euclidean_cluster(cloud, radius, MIN_CLUSTER_SIZE = 1, mode = "cartesian"):
    C = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c = 'r', marker = 'o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    if (mode == "spherical"):
        z, x, y = cloud_sphr[:, 0]*np.sin(cloud_sphr[:, 2])*np.cos(cloud_sphr[:, 1]), cloud_sphr[:, 0]*np.sin(cloud_sphr[:, 2])*np.sin(cloud_sphr[:, 1]), cloud_sphr[:, 0]*np.cos(cloud_sphr[:, 2])
        cloud = np.array([x, y, z]).T

    # uncomment to do voxel downsampling
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(cloud)
    downsamp = o3d_pcd.voxel_down_sample(0.01)
    o3d.visualization.draw_geometries([downsamp])
    # cloud = np.asarray(downsamp.points)

    Node.unexplored = np.array(cloud)

    kd_tree = Node().make_kdtree(cloud, 0, 3)

    while Node.unexplored.shape[0] != 0:
        next_point = Node.unexplored[0]
        C.append([tuple(next_point)])

        Node.unexplored = Node.unexplored[1:]
        Node.unexplored = kd_tree.search_tree(kd_tree, next_point, radius, C)

    clusters = np.array([np.array(cluster) for cluster in C], dtype = object)
    return np.array([cluster for cluster in clusters if cluster.shape[0] > MIN_CLUSTER_SIZE], dtype = object)

def display_clusters(clusters):
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, _ in enumerate(clusters):
        ax.scatter(np.array(clusters[i])[:, 0], np.array(clusters[i])[:, 1], np.array(clusters[i])[:, 2], c = colors[i], marker = 'o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def pca (data):
    means = np.mean(data, axis = 0)
    var = np.var(data, axis = 0)
    centered_data = (data - means) / np.sqrt(var)

    cov = np.cov(centered_data, rowvar = False)
    eigval, eigvec = np.linalg.eig(cov)

    sorted_indices = np.argsort(eigval)[::-1]
    eigval = eigval[sorted_indices]
    # print(eigval)
    eigvec = eigvec[:, sorted_indices]

    xmin, xmax = np.min(centered_data[:, 0]), np.max(centered_data[:, 0])
    ymin, ymax = np.min(centered_data[:, 1]), np.max(centered_data[:, 1])
    zmin, zmax = np.min(centered_data[:, 2]), np.max(centered_data[:, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data[:,0], data[:,1], data[:,2], label="original data")
    ax.scatter(centered_data[:,0], centered_data[:,1], centered_data[:,2], label="centered data")
    # ax.legend()
    # eigen basis
    # aligned_coords = np.matmul(eigvec.T, centered_data)
    # xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])
    # ax.scatter(aligned_coords[:,0], aligned_coords[:,1], aligned_coords[:,2], color='g', label="rotated/aligned data")
    # ax.legend()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # cartesian basis
    # ax.plot([0, 1],  [0, 0], [0, 0], color='b', linewidth=4)
    # ax.plot([0, 0],  [0, 1], [0, 0], color='b', linewidth=4)
    # ax.plot([0, 0],  [0, 0], [0, 1], color='b', linewidth=4)
    # eigen basis

    ax.plot([0, eigvec[0,:][0]], [0, eigvec[0,:][1]], [0, eigvec[0,:][2]], color='r', linewidth=4)
    ax.plot([0, eigvec[1,:][0]],  [0, eigvec[1,:][1]], [0, eigvec[1,:][2]], color='g', linewidth=4)
    ax.plot([0, eigvec[2,:][0]],  [0, eigvec[2,:][1]], [0, eigvec[2,:][2]], color='b', linewidth=4)
    plt.show()

    # rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
    #                                                     [y1, y2, y2, y1, y1, y2, y2, y1],
    #                                                     [z1, z1, z1, z1, z2, z2, z2, z2]])

    # realigned_coords = np.matmul(eigvec, aligned_coords)
    # realigned_coords += means

    # rrc = np.matmul(eigvec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    # rrc += means
    # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(realigned_coords[0,:], realigned_coords[1,:], realigned_coords[2,:], label="rotation and translation undone")
    # # ax.legend()

    # ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color='b')
    # ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color='b')
    # ax.plot(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4], color='b')
    # ax.plot(rrc[0, [3,0]], rrc[1, [3,0]], rrc[2, [3,0]], color='b')

    # # # z2 plane boundary
    # # ax.plot(rrc[0, 4:6], rrc[1, 4:6], rrc[2, 4:6], color='b')
    # # ax.plot(rrc[0, 5:7], rrc[1, 5:7], rrc[2, 5:7], color='b')
    # # ax.plot(rrc[0, 6:], rrc[1, 6:], rrc[2, 6:], color='b')
    # # ax.plot(rrc[0, [7, 4]], rrc[1, [7, 4]], rrc[2, [7, 4]], color='b')

    # # # z1 and z2 connecting boundaries
    # ax.plot(rrc[0, [0, 4]], rrc[1, [0, 4]], rrc[2, [0, 4]], color='b')
    # ax.plot(rrc[0, [1, 5]], rrc[1, [1, 5]], rrc[2, [1, 5]], color='b')
    # ax.plot(rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]], color='b')
    # ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]], color='b')

    # plt.show()
#    return img

# image = cv2.imread("left0.png")
# euclidian_cluster(cloud_sphr, image)
# proj_img = project_to_2d(cloud_sphr, image)
# cv2.imwrite("flatten.png", proj_img)

# uncomment to use test point cloud
# cloud_sphr = np.array([[0.2,8.3,-0.5],[1.1,9,0],[3,7,-1],[2,8.12,0],[1.75,6.5,0.3],
#                       [2,3,0],[4,1,-1],[4,4,-1],[6,1,0],[7,0,-1],[3,2.75,-1.5],[5,2.5,-1.3],
#                       [8,8,-0.5],[7,9,1],[9,6,-1],[6,8,2],[5.5,10,2.5],[5,7,3],[4,7.5,3.5],[5,9,1],[4.5,9.3,1.2],[7,7,2],
#                       [0,8,4],[1,9,4.2],[0.5,7,3.5],[1,8,3.7],[0.5,8.5,3.2]])

# time = timeit.timeit(lambda: euclidean_cluster(cloud = cloud_sphr, radius = 0.15, mode = "spherical"), number = 1) 
C = euclidean_cluster(cloud = cloud_sphr, radius = 0.15, MIN_CLUSTER_SIZE = 10, mode = "spherical")
# display_clusters(C)
# pca(C[1])

# print(f"{time/100} seconds")
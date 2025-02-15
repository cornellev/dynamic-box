import cv2
import os
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import open3d as o3d
import timeit

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

# cloud_sphr = cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(cloud_sphr[:, 0]*np.sin(cloud_sphr[:, 2])*np.cos(cloud_sphr[:, 1]), 
#         cloud_sphr[:, 0]*np.sin(cloud_sphr[:, 2])*np.sin(cloud_sphr[:, 1]), 
#         cloud_sphr[:, 0]*np.cos(cloud_sphr[:, 2]), c = 'r', marker = 'o')
# # ax.scatter(cloud_cyl[:, 0]*np.cos(cloud_cyl[:, 1]), 
# #            cloud_cyl[:, 0]*np.sin(cloud_cyl[:, 1]), 
# #            cloud_cyl[:, 2], c = 'r', marker = 'o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

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

    # def add(self, point):

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
    
    # def traverse_tree (self):
    #     # TODO:
    #     return 0

# Takes in [cloud] in spherical coordinates.
def euclidian_cluster (cloud, image):
    # 0) initialize empty array of clusters C
    C = np.array([])

    # 1) convert spherical to cartesian
    z, x, y = cloud[:, 0]*np.sin(cloud[:, 2])*np.cos(cloud[:, 1]), cloud[:, 0]*np.sin(cloud[:, 2])*np.sin(cloud[:, 1]), cloud[:, 0]*np.cos(cloud[:, 2])
    cloud = np.array([x, y, z]).T

    # 2) downsample point cloud with voxel grid
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(cloud)
    downsamp = o3d_pcd.voxel_down_sample(0.1)
    cloud = np.asarray(downsamp.points)
    
    # 2.5) uncomment to visualize voxel grid
    # o3d.visualization.draw_geometries([downsamp])

    # 3) KD tree: 
    point_tree = Node()
    kd_tree = point_tree.make_kdtree(cloud, 0, 2)

    # 4) Initialize and create clusters: the number of clusters is given by # bounding boxes from YOLO

    return kd_tree


image = cv2.imread("left0.png")
# euclidian_cluster(cloud_sphr, image)
# proj_img = project_to_2d(cloud_sphr, image)
# cv2.imwrite("flatten.png", proj_img)


# uncomment to test 2d point cloud
# cloud = np.array([[1,9],[2,3],[4,1],[3,7],[5,4],[6,8],[7,2],[8,8],[7,9],[9,6]])

# point_tree = Node()
# time = timeit.timeit(lambda: point_tree.make_kdtree(cloud, 0, 3), number = 1)
# time = timeit.timeit(lambda: euclidian_cluster(cloud, image), number = 100)

kd_root = euclidian_cluster(cloud_sphr, image)
split_axis = np.median(kd_root.data[:, kd_root.axis])
# print(f"{time/100} seconds")
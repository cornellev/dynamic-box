import cv2
import os
import numpy as np
import math
import json

# Load JSON file
with open("point_cloud.json", "r") as f:
    data = json.load(f)

# Convert to NumPy array
cloud = np.array(data["points"])


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

Baseline=119.905
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

RT = np.vstack((np.hstack((R, [[Baseline], [TY], [TZ]])), [0,0,0,1]))

def projectTo2D (cloud, image):
    points = []
    for point in cloud:
        x, y, z = point
        points.append([x,y,z,1])

    points = np.array(points).T
    M_intL = np.append(K_L, np.array([[0], [0], [0]]), axis = 1)
    M_intR = np.append(K_R, np.array([[0], [0], [0]]), axis = 1)

    M_L = np.array([[M_intL[0][0], M_intL[0][1], M_intL[0][2], M_intL[0][3]],
                    [M_intL[1][0], M_intL[1][1], M_intL[1][2], M_intL[1][3]],
                    [M_intL[2][0], M_intL[2][1], M_intL[2][2], M_intL[2][3]]])
    M_R = np.array([[M_intR[0][0], M_intR[0][1], M_intR[0][2], M_intR[0][3]],
                    [M_intR[1][0], M_intR[1][1], M_intR[1][2], M_intR[1][3]],
                    [M_intR[2][0], M_intR[2][1], M_intR[2][2], M_intR[2][3]]])
    pt = np.array([x, y, z, 1])

    points_camera = np.dot(M_L, points)

    valid_indices = points_camera[2,:] > 0
    points_camera = points_camera[:, valid_indices]

    points_2d = np.dot(K_L, points_camera[:3, :])

    points_2d /= points_2d[2, :]
    
    for i in range(points_2d.shape[1]):
        u, v = int(points_2d[0, i]), int(points_2d[1, i])
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:  # Check if inside image bounds
            cv2.circle(image, (u, v), 2, (0, 255, 0), -1)  # Draw green dot

    print(points_2d)
    return image

F = np.linalg.inv(K_L).T @ E @ np.linalg.inv(K_R)
E = K_R.T @ F @ K_L
U, S, V = np.linalg.svd(E)

image = cv2.imread("left0.png")
proj_img = projectTo2D(cloud, image)
cv2.imshow("Projected Lidar Points", proj_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

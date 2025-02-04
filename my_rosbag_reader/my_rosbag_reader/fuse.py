import cv2
import os
import numpy as np
import math
import json
import matplotlib.pyplot as plt

# THIS IS ALL WRONG: Z IS NOT DEPTH, NEED RADIUS FOR DEPTH

# Load JSON file
with open("point_cloud.json", "r") as f:
    data = json.load(f)

# Convert to NumPy array
cloud = np.array(data["points"])

cloud_rad = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2 + cloud[:, 2]**2)
cloud_theta = np.arctan(cloud[:, 1]/cloud[:, 0])
cloud_phi = np.arccos(cloud[:, 2]/cloud_rad)

# Point cloud in spherical coordinates, rad = depth.
cloud_sphr = np.array([cloud_rad, cloud_theta, cloud_phi]).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], c = 'r', marker = 'o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

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

    points_2d = np.dot(M_L, points)
    with open("image_cloud.json", "w") as f:
                json.dump({"x":points_2d[0].tolist(),
                           "y":points_2d[1].tolist(),
                           "z":points_2d[2].tolist()}, f)
    
    depth = 255 * (points_2d[2] - np.min(points_2d[2])) / (np.max(points_2d[2]) - np.min(points_2d[2])) 
    depth = np.clip(depth, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    for i, color in zip(range(points_2d.shape[1]), depth_color):
        u, v = int(points_2d[0, i]), int(points_2d[1, i])
        cv2.circle(image, (u, v), 2, color[0].tolist(), -1)  # Draw green dot

    print(points_2d)
    return image

def projectTo2DB (cloud, image):
    d = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0.0],
                          [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.0],
                          [7.402527e-03, 4.351614e-03, 9.999631e-01, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])
    for point in cloud:
        x, y, z = point

        if point[0] > 25.0 or point[0] < 0.0 or abs(point[1]) > 6.0 or point[2] < -1.4:
            continue
        
        X = np.array([point[0], point[1], point[2], 1]).reshape((4, 1))
        Y = np.dot(np.dot(np.dot(np.hstack((K_L, [[0],[0],[0]])), d), RT), X)
        pt = (int(Y[0, 0] / Y[2, 0]), int(Y[1, 0] / Y[2, 0]))

        val = point[0]
        max_val = 20.0
        red = min(255, int(255 * abs((val - max_val) / max_val)))
        green = min(255, int(255 * (1 - abs((val - max_val) / max_val))))
        
        cv2.circle(image, pt, 5, (0, green, red), -1)

    # Display the image
    cv2.imshow("LiDAR data on image overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



F = np.linalg.inv(K_L).T @ E @ np.linalg.inv(K_R)
E = K_R.T @ F @ K_L
U, S, V = np.linalg.svd(E)

image = cv2.imread("left0.png")
proj_img = projectTo2D(cloud, image)
cv2.imshow("Projected Lidar Points", proj_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

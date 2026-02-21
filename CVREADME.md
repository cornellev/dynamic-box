## CV Yap: Given Stereo Images, How Do We Get a Depth Map?
1) Used YOLOv3 to generate 2D bounding boxes for objects in stereo images taken by the ZED camera. Returns 2D bounding box information in the form of $(x_l,y_l,w,h)$.

2) We cannot assume that our intrinsic matrix $K$ is identity $I$, so we need to derive fundamental matrix $F$ to relate left and right images together. There's definitely an OpenCV function for this, but the high level way to get $F$ is to first >8 corresponding points between the two images, construct magical matrix $A$, then get the eigenvector corresponding to the smallest eigenvalue of $A^TA$. 

   Ask me if you want the deets.
   
   Q: How do I get these >8 corresponding points?

   A: SIFT feature descriptor generation and then feature matching (SAD, cosine-similarity, etc.). Or if you really trust that for each object, YOLO outputs accurate bounding boxes in both images, then you can also use vertices of the boxes as corresponding features.
3) If you don't want to estimate $F$ from the 8-point algorithm, here's another way. 

   By convention, we let the left camera's coordinate frame be aligned with the world coordinate frame. This means $c_l = [R | t] w_l = w_l$ and projection matrix from world to left camera frame is just $I$. Then we can compute the epipole $e_R$ of the right image with $F^Te_R=0$, where $e_R$ is in the null space of $F$. Finally, we derive the right camera's projection matrix as $K_R=[[e_R]_xF+e_2v^T]$.

   But like seriously, we are already given $K$ from the ZED camera specifications.

4) For funsies, $F = K_R^{-T} \cdot [t]_x \cdot R \cdot K_L^{-1} = K_R^{-T} \cdot E \cdot K_L^{-1}$, where essential matrix $E = [t]_x \cdot R$. 

   Q: What's an essential matrix?

   A: Well, in a happy world, $K_L = K_R = I$, and we can relate the left and right images by $E$. 

   Use $F$, $K_R$ and $K_L$ to derive essential matrix $E={K_R^T}\cdot{F}\cdot{K_L}$. 
   
5) So I claimed that we could use $F$ to relate two images. What $F$ does is, given a point $x_{left}$ in the left image, we can compute the right image's epipolar line $l={F}\cdot x_{left}$. 

   Q: Wow, amazing. What is the point of getting this epipolar line?

   A: Theoretically, the point $x_{right}$ in the right image that corresponds to $x_{left}$ is along epipolar line $l$. So we just have to search along $l$ for the window most similar to $x_{left}$ to find $x_{right}$. Once again, use some similarity metric to achieve this. 
   
6) Given corresponding left and right points, we can convert 2D points in the images to 3D points by calculating the depth of that point from the camera. We use the formula $Z=\dfrac{{f}\cdot{B}}{d}$, where $Z$ is distance of the object from the camera, $f$ is focal length, $B=120mm$ (for ZED) is baseline, or the distance between the two cameras, and $d$ is disparity (horizontal pixel distance between $x_{left}$ and $x_{right}$).
   
5) Note that $x_{left}$ and $x_{right}$ are both pixel coordinates, to get their $X$ and $Y$ coordinates in world frame, we use projection equations: $X={x-c_x}\cdot{\dfrac{Z}{f_x}}$, $Y={y-c_y}\cdot{\dfrac{Z}{f_y}}$, and $Z=depth(x,y)$ where $(x,y)$ is the pixel coordinates of the corners of the 2D bounding box, focal lengths of the left ZED camera $f_x=700.819$ and $f_y=700.819$.

6) For each object detected by YOLOv3, convert 2D points constrained in its corresponding bounding box into a 3D point cloud. Then find the eigenvalues and eigenvectors of the covariance matrix of the 3D dataset to construct a rotation matrix, which is used to derive yaw, pitch, and roll of the object. The dimensions of the 3D bounding box is constrained within the 2D bounding box, and orientation and pose are estimated using the calculated yaw, pitch, and roll.


## Important links:

[An Introduction to 3D Computer Vision Techniques and Algorithms](https://ia801208.us.archive.org/12/items/an-introduction-to-3-d-computer-vision-techniques-and-algorithms-cyganek-siebert-2009-02-09/An%20Introduction%20to%203D%20Computer%20Vision%20Techniques%20and%20Algorithms%20%5BCyganek%20%26%20Siebert%202009-02-09%5D.pdf)

[ZED Calibration File](https://support.stereolabs.com/hc/en-us/articles/360007497173-What-is-the-calibration-file)

[CS231A Course Notes 3: Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)

[Epipolar Geometry and the Fundamental Matrix](https://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf)

[Complex YOLO: YOLOv4 for 3D Object Detection](https://medium.com/@mohit_gaikwad/complex-yolo-yolov4-for-3d-object-detection-3c9746281cd2)

[Stereo R-CNN based 3D Object Detection](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN?tab=readme-ov-file)

[3D Reconstruction and Epipolar Geometry](https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry)

[The 8-point algorithm](https://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf)


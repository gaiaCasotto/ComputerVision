import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from util_functions import pi, invPi, camera_intrinsic, projectpoints, crossProduct, essential_matrix, fundamental_matrix, DrawLine, triangulate



#main
plt.ion()    

K = camera_intrinsic(1000, (300, 200))
R1 = np.eye(3)
t1 = np.array([[0, 0, 0]]).T
t2 = np.array([[0.2, 2, 1]]).T
R2 = Rotation.from_euler('xyz', [0.7, -0.5, 0.8]).as_matrix()

#part 3.1
Q = np.array([[1, 0.5, 4, 1]]).T 
Q = invPi(Q) #make 3x1 matrix
print(Q)
cam_pos1 = np.hstack((R1, t1))
cam_pos2 = np.hstack((R2, t2))
q1 = projectpoints(K, cam_pos1, Q)
q2 = projectpoints(K, cam_pos2, Q)
print(f'q1: {q1.T}')
print(f'q2: {q2.T}')

#part 3.2
#part 3.3
F = fundamental_matrix(K, R1, t1, K, R2, t2)
print(f"part 3.3 -> F: {F}")

#part 3.4
l2 = F @ pi(q1)      # epipolar line of q1 in camera 2
l2 = l2/l2[2]*-5.285         # to match the answer
print(f'part 3.4 ->l2: {l2}')

# part 3.5
# Check if q2 is located on the epipolar line l2
print(l2.shape)
print(q1.shape)

point = pi(q2).T @ l2   
print(f'part 3.5 : point -> {point}')             # should be 0
# Since the dot product is close to 0, the point is on the line.

#part 3.6 -> written
#part 3.7 -> written

#part 3.8
data = np.load('TwoImageDataCar.npy', allow_pickle=True).item()
print(data.keys())
#compute the fundamental matrix between the two images
im1 = data['im1']
im2 = data['im2']
K = data['K']
t1, t2 = data['t1'], data['t2']
R1, R2 = data['R1'], data['R2']
F = fundamental_matrix(K, R1, t1, K, R2, t2)
print(f'part 3.8 -> F: {F}')

'''plt.subplot(1,2,1)
plt.imshow(im1)
plt.subplot(1,2,2)
plt.imshow(im2)
plt.show()'''

#part 3.9  FIX!!!
# Click on a point, and draw the epipolar line on the other image
figure = plt.figure()
ax1 = plt.subplot(1,2,1)
ax1.imshow(im1)
ax1.set_title('Click on a point')
ax2 = plt.subplot(1,2,2)
ax2.imshow(im2)
ax2.set_title('Epipolar line will be drawn here')

# Click on a point in the first image
q = plt.ginput(1)
q = q[0]
ax1.plot(q[0], q[1], 'bx')

l = F @ pi(np.array([q]).T)
DrawLine(l, im2.shape)
plt.show()


# Ex 3.10    FIX
# Flip images
figure = plt.figure()
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(im2)
ax1.set_title("Click on a point")
ax2 = plt.subplot(1, 2, 2)
ax2.imshow(im1)
ax2.set_title("Epipolar line will be drawn here")

# Click on a point in the first image
q = plt.ginput(1)
q = q[0]
ax1.plot(q[0], q[1], "bx")

l = F @ pi(np.array([q]).T)
DrawLine(l, im1.shape)

#part 3.11
# Camera parameters
K = camera_intrinsic(1000, (300, 200))
R1 = np.eye(3)
t1 = np.array([[2, 3, 4]]).T
R2 = Rotation.from_euler('xyz', [0.7, -0.5, 0.8]).as_matrix()
t2 = np.array([[0.2, 2, 1]]).T
cam_pos1 = np.hstack((R1, t1))
cam_pos2 = np.hstack((R2, t2))

# Arbitrary 3D point
Q = np.array([[1, 0.5, 4]]).T

# Triangulate procedure
q1 = projectpoints(K, cam_pos1, Q)  # project Q to image plane of camera 1
q2 = projectpoints(K, cam_pos2, Q)  # project Q to image plane of camera 2
P1 = K @ cam_pos1
P2 = K @ cam_pos2
Q = triangulate([q1, q2], [P1, P2])
print(f"pt 3.11 -> Q is {Q}")



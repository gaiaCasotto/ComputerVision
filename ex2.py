import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from util_functions import box3d, pi, invPi, projectpoints, projectpoints_dist, distort, undistort_image, camera_intrinsic, hest, normalize2d, hest_from_image


#main
image_name = 'gopro_robot.jpeg'
gopro_im = cv.imread(image_name)[:, :, ::-1]
gopro_im = gopro_im.astype(float)/255

#part 2.3 
im_height, im_width, _ = gopro_im.shape
f = 0.455732 * im_width
distCoeff = [-0.245031, 0.071524, -0.00994978]
alpha = 1
beta  = 0
deltax, deltay = 400, 400
#pp is probably in the middle of the image
pp = [ (im_height//2), (im_width//2)]
'''K = [[f, 0, pp[0]],
     [0, f, pp[1]],
     [0, 0, 1    ]]'''
K = camera_intrinsic(f, (deltax, deltay), alpha, beta)
gopro_K = camera_intrinsic(f, (gopro_im.shape[1] / 2, gopro_im.shape[0] / 2), alpha, beta)
cam_pos = np.array([[1, 0, 0, 0  ],
                    [0, 1, 0, 0.2],
                    [0, 0, 1, 1.5]
                   ])
R = np.eye(3)
t = np.array([[0, 0.2, 1.5]])
box = box3d()
#print(K)

#part 2.4 
P = projectpoints_dist(K, cam_pos, box, distCoeff)
plt.scatter(P[0, :], P[1, :])
plt.savefig("distorted_box3d.png")
plt.show()

image = cv.imread("distorted_box3d.png")
undistorted_im = undistort_image(image, K, distCoeff)
plt.imshow(undistorted_im)


# Display the original and undistorted images side by side
plt.figure(figsize=(12, 6))
# Original image

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(gopro_im)
plt.axis('off')

# Undistorted image
undistorted_gopro = undistort_image(gopro_im, gopro_K, distCoeff)
plt.subplot(1, 2, 2)
plt.title('Undistorted Image')
plt.imshow(undistorted_gopro)
plt.axis('off')
plt.show()

#part 2.5
p2a = np.array([1, 1]).reshape(-1, 1)
p2b = np.array([0, 3]).reshape(-1, 1)
p2c = np.array([2, 3]).reshape(-1, 1)
p2d = np.array([2, 4]).reshape(-1, 1)
H = np.array([[-2, 0, 1], [1, -2, 0], [0, 0, 3]]) #3x3
p2 = np.hstack((p2a, p2b, p2c, p2d))
# Apply homography to 2D points
invpi = pi(p2) 
qh = H @ invpi # 3 x n
q = invPi(qh)  # 2 x n
print("part 2.5")
print(f"q is {q}\n")

#part 2.6
H = hest(q, p2)
print("part 2.6")
with np.printoptions(precision = 3, suppress = True):
    print(H) # If the input points are flipped, inv(H) will be obtained.
#now checking with points from 2.5:
H = np.array([[-2, 0, 1], [1, -2, 0], [0, 0, 3]])
norm = np.linalg.norm(H, "fro")
res = H / norm
print(res)
print()

#part 2.7
print("part 2.7")
qn, T = normalize2d(p2)
print(f"qn is {qn}")
print(f"T is {T}\n")

#part 2.8
print("part 2.8")
H = hest(q, p2)
with np.printoptions(precision=3, suppress=True):
    print(f"H is {H}\n")

#part 2.9
print("part 2.9")
# generate 100 random 2D points and random H. use hest to estimate H
q2 = np.random.randn(2, 100)
q2h = np.vstack((q2, np.ones((1, 100))))
H_true = np.random.randn(3, 3)
q1h = H_true @ q2h
q1 = invPi(q1h)

H_est = hest(q1, q2)
print(H_true / np.linalg.norm(H_true, "fro"))
print(H_est)



#part 2.1
box = box3d()
K = np.eye(3) #creates identity matrix
f = 600
alpha = 1
beta  = 0
delta_x, delta_y = 400, 400
K[0][0] = f
K[1][1] = f
K[0][2] = K[1][2] = delta_x # = delta_y
'''resolution = a, the principal point is exactly in the middle of the sensor. So, for this
camera the sensor has 2 x 400 = 800 pixels along each dimension i.e. a resolution of 800 x 800
pixels.'''
#t = [0, .2, 1.5]
cam_pos = np.array([[1, 0, 0, 0  ],
                    [0, 1, 0, 0.2],
                    [0, 0, 1, 1.5]
                   ])

distCoeff = [-0.2]

output = projectpoints_dist(K, cam_pos, box, distCoeff)
print(f"projection1: {output}" )

outputx = output[0]
outputy = output[1]

print()
plt.scatter(outputx, outputy)
plt.show() #pt 2.2




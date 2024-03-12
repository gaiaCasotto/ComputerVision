import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
from util_functions import box3d, pi, invPi
#print(cv.__version__)

#FIX!!!
'''
def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

def pi(coords):
    print(f"before : {coords.shape}")
    print(coords)
    ones = np.ones(240, dtype=int)
    print(f"ones: {ones.shape}")
    print(f"ones: {ones}")
    return np.vstack((coords, ones))
    

def invPi(hom_coords):
    hom_coords = hom_coords[:-1]/hom_coords[-1]
    return hom_coords 
'''
def projectpoints(K, cam_pos, Q):
    hom_proj = []
    proj     = []
    #Q = 3xn (3x241)
    #K = 3x3
    #ph = K@cam_pos@ = k @ cam_pos @ Q[i]
    Q_hom = pi(Q)
    #print(Q_hom)
    for i in range(240):
        q = Q_hom[:, i]
        hom_p =  K @ cam_pos @ q
        hom_proj.append(hom_p)
    proj = invPi(hom_proj)
    return proj


'''
def projectpoints(K, cam_pos, Q):  correct version
    x = []
    y = []
    w = []
    proj     = []
    #Q = 3xn (3x241)
    #K = 3x3
    #ph = K@cam_pos@ = k @ cam_pos @ Q[i]
    Q_hom = pi(Q)
    #print(Q_hom)
    for i in range(len(Q[0])):
        q = Q_hom[:, i]
        hom_p =  K @ cam_pos @ q
        x.append(hom_p[0])
        y.append(hom_p[1])
        w.append(hom_p[2])
    hom_proj = np.vstack( (x,y,w) )
    proj = invPi(hom_proj)
    return proj

'''

image_name = 'gaiaFace.jpg'
image      = cv.imread(image_name)
#cv.imshow("Image", image)
#cv.waitKey(0)
#cv.destroyAllWindows()
box = box3d()
print(box.shape)
#print(f"box", box)
Q = np.array([[1, 2, 3],
              [5, 6, 7],
              [9, 10, 11]])
K = np.eye(3) #creates identity matrix
cam_pos = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 4]
                   ])

projection1 = projectpoints(K, cam_pos, box)
print(f"projection1: {projection1}" )

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.grid()

ax.set_title('3D Scatter Plot')

# Plot the points
ax.scatter(projection1[0], projection1[1], projection1[2], c='b')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

cam_pos2 = np.array([[math.cos(30), 0, math.sin(30), 0],
                    [0, 1, 0, 0],
                    [-math.sin(30), 0, math.cos(30), 4]
                   ])
box2 = box3d()
projection2 = projectpoints(K, cam_pos2, box2)
#print(f"projection2: {projection2}" )
#print("end 2")



import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt
import itertools as it

def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2

def pi(coords):
    ones = np.ones(len(coords[0]), dtype=int)
    return np.vstack((coords, ones))
    

def invPi(hom_coords):
    hom_coords = hom_coords[:-1]/hom_coords[-1]
    return hom_coords 
'''
def projectpoints(K, cam_pos, Q):
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

def projectpoints(K, cam_pos, Q, distCoeffs=[]):
    """
    Project 3D points in Q onto a 2D plane of a camera with distortion.

    Args:
        K : 3 x 3, intrinsic camera matrix
        cam_pos : 3 x 4, camera Rotation and Translation matrix
        Q: 3 x n, 3D points matrix
        distCoeffs: [k3,k5,k7,...] distortion coefficients

    Returns:
        P : 2 x n, 2D points matrix
    """
    if Q.shape[0] != 3:
        raise ValueError("Q must be 3 x n")
    if K.shape != (3, 3):
        raise ValueError("K must be 3 x 3")
    if cam_pos.shape != (3, 4):
        raise ValueError("R must be 3 x 4")

    Qh = pi(Q)  # 4 x n
    qh = cam_pos @ Qh  # 3 x n
    q = invPi(qh)  # 2 x n
    qd = distort(q, distCoeffs)  # 2 x n
    Ph = K @ pi(qd)  # 3 x n
    P = invPi(Ph)  # 2 x n
    return P


def projectpoints_dist(K, cam_pos, Q, distCoeff):
    if(len(distCoeff) == 0):
        print("need more coeffs")
        return None
    x = []
    y = []
    w = []
    proj = []
    #Q = 3xn (3x241)
    #K = 3x3
    #ph = K@cam_pos@ = k @ cam_pos @ Q[i]
    Q_hom = pi(Q)
    for i in range(len(Q[0])):
        q = Q_hom[:, i]
        q[0], q[1] = distort(q[0], q[1], distCoeff)
        hom_p =  K @ cam_pos @ q
        x.append(hom_p[0])
        y.append(hom_p[1])
        w.append(hom_p[2])
    hom_proj = np.vstack( (x,y,w) )
    proj = invPi(hom_proj)
    return proj

#github: yufanana 

def distort(q, distCoeff): #part2.2
    r = np.sqrt((q[0]) ** 2 + (q[1]) ** 2)
    correction = 1
    for i in range(len(distCoeff)):
        exp = 2 * (i + 1)
        correction += distCoeff[i] * r**exp
    qd = q * correction
    return qd

def undistort_image(image, K, distCoeff): #part 2.4
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0])) #just puts all x in one array and all y in another array
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1) #stacks them so that they are homogenous
    
    q = pi(np.linalg.inv(K) @ p)
    for i in range(len(q[0])):
        q[0][i], q[1][i] = distort(q[0][i], q[1][i], distCoeff)
    q_d = invPi(q)
    p_d = K @ q_d
    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    assert (p_d[2]==1).all(), 'You did a mistake somewhere'
    im_undistorted = cv.remap(image, x_d, y_d, cv.INTER_LINEAR)
    return im_undistorted



def camera_intrinsic(f, c, alpha=1, beta=0):
    """
    Create a camera intrinsic matrix

    f : float, focal length
    c : 2D principal point
    alpha : float, skew
    beta : float, aspect ratio
    """
    K = np.array([[f, beta * f, c[0]], [0, alpha * f, c[1]], [0, 0, 1]])
    return K


#part 2.6
def hest(q1, q2):
    """
    Calculate the homography matrix from n sets of 2D points
    q1 : 2 x n, 2D points in the first image
    q2 : 2 x n, 2D points in the second image
    H : 3 x 3, homography matrix
    """
    n = q1.shape[1]
    B = []
    for i in range(n):
        x1, y1 = q1[:, i]
        x2, y2 = q2[:, i]
        Bi = np.array(
            [
                [0, -x2, x2 * y1, 0, -y2, y2 * y1, 0, -1, y1],
                [x2, 0, -x2 * x1, y2, 0, -y2 * x1, 1, 0, -x1],
                [-x2 * y1, x2 * x1, 0, -y2 * y1, y2 * x1, 0, -y1, x1, 0],
            ]
        )
        B.append(Bi)
    B = np.array(B).reshape(-1, 9)
    U, S, Vt = np.linalg.svd(B)
    H = Vt[-1].reshape(3, 3)
    return H.T

#part 2.7
def normalize2d(q):
    """
    Normalize 2D points.
    
    q : 2 x n, 2D points
    qn : 2 x n, normalized 2D points
    """
    if q.shape[0] != 2:
        raise ValueError("q must have 2 rows")
    if q.shape[1] < 2:
        raise ValueError("At least 2 points are required to normalize")

    mu = np.mean(q, axis=1).reshape(-1, 1)
    mu_x = mu[0].item()
    mu_y = mu[1].item()
    std = np.std(q, axis=1).reshape(-1, 1)
    std_x = std[0].item()
    std_y = std[1].item()
    Tinv = np.array([[std_x, 0, mu_x], [0, std_y, mu_y], [0, 0, 1]])
    T = np.linalg.inv(Tinv)
    qn = T @ pi(q)
    qn = invPi(qn)
    return qn, T

#part 2.8 : include normalization in the homography estimation
def hest(q1, q2, normalize=False):
    """
    Calculate the homography matrix from n sets of 2D points
    q1 : 2 x n, 2D points in the first image
    q2 : 2 x n, 2D points in the second image
    H : 3 x 3, homography matrix
    """
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("Number of points in q1 and q2 must be equal")
    if q1.shape[1] < 4:
        raise ValueError("At least 4 points are required to estimate a homography")
    if q1.shape[0] != 2 or q2.shape[0] != 2:
        raise ValueError("q1 and q2 must have 2 rows")

    if normalize:
        q1, T1 = normalize2d(q1)
        q2, T2 = normalize2d(q2)
    n = q1.shape[1]
    B = []
    for i in range(n):
        x1, y1 = q1[:, i]
        x2, y2 = q2[:, i]
        Bi = np.array(
            [
                [0, -x2, x2 * y1, 0, -y2, y2 * y1, 0, -1, y1],
                [x2, 0, -x2 * x1, y2, 0, -y2 * x1, 1, 0, -x1],
                [-x2 * y1, x2 * x1, 0, -y2 * y1, y2 * x1, 0, -y1, x1, 0],
            ]
        )
        B.append(Bi)
    B = np.array(B).reshape(-1, 9)
    U, S, Vt = np.linalg.svd(B)
    H = Vt[-1].reshape(3, 3)
    if normalize:
        H = np.linalg.inv(T1) @ H @ T2
    return H

#part 2.10
def hest_from_image(im1, im2, n):
    """
    Estimate homography from n pairs of points in two images
    im1 : np.array, first image
    im2 : np.array, second image
    n : int, number of pairs of points
    H : 3 x 3, homography matrix
    """
    plt.imshow(im1)
    plt.title("Click on points in an ascending order")
    p1 = plt.ginput(n)
    plt.close()
    plt.imshow(im2)
    plt.title("Click on points in an ascending order")
    p2 = plt.ginput(n)
    plt.close()
    H = hest(np.array(p1).T, np.array(p2).T, True)
    return H


#part 3.2
def crossProduct(r):
    if r.shape == (3,1):
        r = r.flatten()
    if r.shape != (3,):
        raise ValueError('r must be a 3x1 vector')
    
    rtn = np.array([ [0, -r[2], r[1]],
                     [r[2], 0, -r[0]],
                     [-r[1], r[0], 0]
                   ])
    return rtn

#part 3.3
def essential_matrix(R,t):
    rtn = crossProduct(t) @ R
    return rtn

def fundamental_matrix(K1, R1, t1, K2, R2, t2):
    '''
    Returns the fundamental matrix, assuming camera 1 coordinates are 
    on top of global coordinates.    
    '''
    if R1.shape != (3,3) or R2.shape != (3,3):
        raise ValueError('R1 or R2 not 3x3 matrix')
    if t1.shape == (3,) or t2.shape == (3,):
        t1 = t1.reshape(-1,1)
        t2 = t2.reshape(-1,1)
    if t1.shape != (3,1) or t2.shape != (3,1):
        raise ValueError('t1 and t2 must be 3x1 matrices')
    if K1.shape != (3,3) or K2.shape != (3,3):
        raise ValueError('K1 and K2 must be 3x3 matrices')
    
    # When {camera1} and {camera2} are not aligned with {global}
    R_tilde = R2 @ R1.T
    t_tilde = t2 - R_tilde @ t1
    print(f't_tilde shape : {t_tilde.shape}')
    E = essential_matrix(R_tilde, t_tilde)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

#part 3.9
# Click on a point, and draw the epipolar line on the other image
def DrawLine(l, shape):
    # Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame

    def in_frame(l_im):
        '''Returns the intersection point of the line with the image frame.'''
        q = np.cross(l.flatten(), l_im) # intersection point
        q = q[:2]/q[2]                  # convert to inhomogeneous
        if all(q>=0) and all(q+1<=shape[1::-1]):
            return q
    
    # 4 edge lines of the image
    lines = [[1, 0, 0],             # x = 0
             [0, 1, 0],             # y = 0
             [1, 0, 1-shape[1]],    # x = shape[1]
             [0, 1, 1-shape[0]]]    # y = shape[0]
    
    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]
    
    if (len(P)==0):
        print("Line is completely outside image")
    plt.plot(*np.array(P).T)
    plt.show()


# Ex 3.11
# Triangulation
def triangulate(q_list, P_list):
    """
    Triangulate a single 3D point seen by n cameras.

    q_list : nx2x1 list of 2x1 pixel points
    P_list : nx3x4, list of 3x4 camera projection matrices
    """
    B = []  # 2n x 4 matrix
    for i in range(len(P_list)):
        qi = q_list[i]
        P_i = P_list[i]
        B.append(P_i[2] * qi[0] - P_i[0])
        B.append(P_i[2] * qi[1] - P_i[1])
    B = np.array(B)
    U, S, Vt = np.linalg.svd(B)
    # TODO: why is the 3D point the last column of the matrix V
    # returned by the SVD, normalized to have a last coordinate of 1.
    Q = Vt[-1, :-1] / Vt[-1, -1]
    return Q

#part 4.1
#creates the projection matrix from camera parameters K,R,t
def projection_matrix(K, R, t):
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix")
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    if t.shape != (3, 1):
        raise ValueError("t must be a 3x1 matrix")

    P = K @ np.hstack((R, t))
    return P

#part 4.2
#uses Q,q to estimate P (projection matrix) with DLT. Points not normalized
def pest(Q, q, normalize=False):
    '''Write a function pest that uses Q and q to estimate P with the DLT.
      Do not normalize your points.
    Use the estimated projection matrix P est to project the points Q,
      giving you the reprojected points q.
    '''
    if Q.shape[0] != 3:
        raise ValueError("Q must be a 3 x n array of 3D points")
    if q.shape[0] != 2:
        raise ValueError("q must be a 2 x n array of 2D points")

    if normalize:
        q, T = normalize2d(q)

    q = pi(q)  # 3 x n
    Q = pi(Q)  # 4 x n
    n = Q.shape[1]  # number of points
    B = []
    for i in range(n):
        Qi = Q[:, i]
        qi = q[:, i]
        # Xi, Yi, Zi = Qi
        # xi, yi, _ = qi
        # Bi = np.array([[0, -Xi, Xi*yi, 0, -Yi, Yi*yi, 0, -Zi, Zi*yi, 0, -1, yi],
        #                [Xi, 0, -Xi*xi, Yi, 0, -Yi*xi, Zi, 0, -Zi*xi, 1, 0, -xi],
        #                [-Xi*yi, Xi*xi, 0, -Yi*yi, Yi*xi, 0, -Zi*yi, Zi*xi, 0, -yi, xi, 0]])
        Bi = np.kron(Qi, crossProduct(qi))
        B.append(Bi)
    B = np.array(B).reshape(3 * n, 12)
    U, S, Vt = np.linalg.svd(B)
    P = Vt[-1].reshape(4, 3)
    P = P.T
    if normalize:
        P = np.linalg.inv(T) @ P
    return P

def compute_rmse(q_true, q_est):
    """
    Returns the root mean square error between the true and estimated 2D points.

    Args:
        q_true: 2 x n array of true 2D points
        q_est: 2 x n array of estimated 2D points
    """
    if q_true.shape[0] != 2 or q_est.shape[0] != 2:
        raise ValueError("q_true and q_est must be 2 in the first dimension")
    if q_true.shape[1] != q_est.shape[1]:
        raise ValueError("q_true and q_est must have the same number of points")
    se = (q_est - q_true) ** 2
    return np.sqrt(np.mean((se)))

#part 4.3
'''
Here we will perform camera calibration with checkerboards. 
We do not yet have the ability to detect checkerboards, so for now we will 
define the points ourselves.
'''
#to define the points
def checkerboard_points(n, m):
    '''returns 3D points 
    The points should be returned as a 3x(n*m)
    matrix and their order does not matter
     These points lie in the z = 0 plane by definition.
    '''
    checkerboard = np.array(
        [
            (i - (n - 1) / 2, j - (m - 1) / 2, 0)
            for i in range(n)
            for j in range(m)
        ]
    ).T
    return checkerboard
#checkerboard is a 3x(n*m) matrix

#part 4.5
def estimateHomographies(Q_omega, qs):
    '''
    Q_omega: an array original un-transformed checkerboard points in 3D, for example Q . 
    qs: a list of arrays, each element in the list containing Q projected to the image plane 
    from different views, for example qs could be [qa, qb, qc ].

    Returns the list of homographies that map from Q_omega to each of the entries in qs.
    '''
    Hs = []
    Q = Q_omega[:2]  # remove 3rd row of zeros
    for q in qs:
        H = hest(Q,q)
        Hs.append(H)
    return Hs

#part 4.6
# Estimate b vector
def form_vi(H, a, b):
    '''
    Form 1x6 vector vi using H and indices alpha, beta.

    Args:
        H : 3x3 homography
        a, b : indices alpha, beta

    Returns:
        vi : 1x6 vector
    '''
    # Use zero-indexing here. Notes uses 1-indexing.
    a = a - 1
    b = b - 1
    vi = np.array(
        [
            H[0, a] * H[0, b],
            H[0, a] * H[1, b] + H[1, a] * H[0, b],
            H[1, a] * H[1, b],
            H[2, a] * H[0, b] + H[0, a] * H[2, b],
            H[2, a] * H[1, b] + H[1, a] * H[2, b],
            H[2, a] * H[2, b],
        ]
    )
    vi = vi.reshape(1, 6)
    return vi
    
def estimate_b(Hs):
    """
    Estimate b matrix used Zhang's method for camera calibration.

    Args:
        Hs : list of 3x3 homographies for each view

    Returns:
        b : 6x1 vector
    """
    V = []  # coefficient matrix
    # Create constraints in matrix form
    for H in Hs:
        vi_11 = form_vi(H, 1, 1)
        vi_12 = form_vi(H, 1, 2)
        vi_22 = form_vi(H, 2, 2)
        v = np.vstack((vi_12, vi_11 - vi_22))  # 2 x 6
        V.append(v)
    # V = np.array(V) creates the wrong array shape
    V = np.vstack(V)  # 2n x 6
    U, S, bt = np.linalg.svd(V.T @ V)
    b = bt[-1].reshape(6, 1)
    return b

def b_from_B(B):
    """
    Returns the 6x1 vector b from the 3x3 matrix B.

    b = [B11 B12 B22 B13 B23 B33].T
    """
    if B.shape != (3, 3):
        raise ValueError("B must be a 3x3 matrix")

    b = np.array((B[0, 0], B[0, 1], B[1, 1], B[0, 2], B[1, 2], B[2, 2]))
    b = b.reshape(6, 1)
    return b

#part 4.7
# Estimate intrinsic matrix using equations from Zhang's paper
def estimate_intrinsics(Hs):
    """
    Estimate intrinsic matrix using Zhang's method for camera calibration.

    Args:
        Hs : list of 3x3 homographies for each view

    Returns:
        K : 3x3 intrinsic matrix
    """
    b = estimate_b(Hs)
    B11, B12, B22, B13, B23, B33 = b
    # Appendix B of Zhang's paper
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = lambda_ * v0 / beta - B13 * alpha**2 / lambda_
    # above values are sequences [value], so using [0] below is needed
    K = np.array([[alpha[0], gamma[0], u0[0]], [0, beta[0], v0[0]], [0, 0, 1]])
    return K
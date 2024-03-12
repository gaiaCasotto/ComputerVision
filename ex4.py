import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from util_functions import (
    camera_intrinsic,
    projection_matrix,
    projectpoints,
    pest,
    pi,
    invPi,
    compute_rmse,
    checkerboard_points,
    estimateHomographies,
    estimate_b,
    b_from_B,
    form_vi,
    estimate_intrinsics,
)

np.set_printoptions(precision=4)




#main
# part 4.1
# Find projection matrix and projections
f = 1000
resolution = (1920, 1080)
principal_point = (resolution[0] / 2, resolution[1] / 2)
R = np.array(
    [
        [np.sqrt(0.5), -np.sqrt(0.5), 0],
        [np.sqrt(0.5), np.sqrt(0.5), 0],
        [0, 0, 1],
    ]
)
t = np.array([[0, 0, 10]]).T
K = camera_intrinsic(f, principal_point)
#calculate projection matrix
P = projection_matrix(K, R, t)

Q = np.array(
    [(x, y, z) for x in [0, 1] for y in [0, 1] for z in [0, 1]]
).T  # 3 x n
cam_pos = np.hstack((R,t))
q = projectpoints(K, cam_pos, Q)
with np.printoptions(precision=3):
    print(f"part 4.1 -> {q}")


#part 4.2
P_est = pest(Q, q)
print(f"part 4.2 -> {P_est}") 
q_est = P_est @ pi(Q)
rmse = compute_rmse(q, invPi(q_est))
print("RMSE: ", rmse)  # should be close to 0

# Normalize points before estimating P
P_est = pest(Q, q, normalize=True)
q_est = P_est @ pi(Q)

rmse = compute_rmse(q, invPi(q_est))
print("RMSE normalized: ", rmse)
# Normalizing the points should improve reprojection error slightly. And it DOES!!

#part 4.3
Qcb = checkerboard_points(5, 4)
print('part 4.3: checkerboard')
print(Qcb.shape)
print(Qcb)

#part 4.4
Ra = Rotation.from_euler("xyz", [ np.pi/10, 0, 0]).as_matrix()
Rb = Rotation.from_euler("xyz", [        0, 0, 0]).as_matrix()
Rc = Rotation.from_euler("xyz", [-np.pi/10, 0, 0]).as_matrix()

Q_omega = checkerboard_points(10, 20)
Qa = Ra @ Q_omega
Qb = Rb @ Q_omega
Qc = Rc @ Q_omega

cam_pos_a = np.hstack((Ra, t))
cam_pos_b = np.hstack((Rb, t))
cam_pos_c = np.hstack((Rc, t))

qa = projectpoints(K, cam_pos_a, Qa)
qb = projectpoints(K, cam_pos_b, Qb)
qc = projectpoints(K, cam_pos_c, Qc)

# Visualize the 3 views
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(Qa[0, :], Qa[1, :], Qa[2, :])
ax.scatter(Qb[0, :], Qb[1, :], Qb[2, :])
ax.scatter(Qc[0, :], Qc[1, :], Qc[2, :])
plt.show()

#part 4.5
boards = [qa, qb, qc]
Hs = estimateHomographies(Q_omega, boards)

Q_omega_h = pi(Q_omega[:2])  # Q_omega without z, in homogenous
qa_est = Hs[0] @ Q_omega_h
qb_est = Hs[1] @ Q_omega_h
qc_est = Hs[2] @ Q_omega_h

qa_est = qa_est[:2] / qa_est[-1]
qb_est = qb_est[:2] / qb_est[-1]
qc_est = qc_est[:2] / qc_est[-1]

# Compute reprojection error
print("RMSE: ", compute_rmse(qa, qa_est))
print("RMSE: ", compute_rmse(qb, qb_est))
print("RMSE: ", compute_rmse(qc, qc_est))

#RMSE:  883.1084377261193
#RMSE:  902.2225410500663
#RMSE:  998.7673644727909    //These are very very big??? WHYYYYY???

#part 4.6
print("part 4.6 -> \n")
b = estimate_b(Hs)
B_true = np.linalg.inv(K.T) @ np.linalg.inv(K)
b_true = b_from_B(B_true)

print("b_est:\n", b / np.linalg.norm(b))
print("b_true:\n", b_true / np.linalg.norm(b_true))

# Check: v11 b_true == h1.T B_true h1, first -> zero indexing
h1 = Hs[0][:, 0]
v11 = form_vi(Hs[0], 1, 1)

print(v11 @ b_true)
print(h1.T @ B_true @ h1)
# The 2 values should be the same.

# Check: v22 b_true == h1.T B_true h2
h2 = Hs[0][:, 1]
v22 = form_vi(Hs[0], 2, 2)

print(v22 @ b_true)
print(h2.T @ B_true @ h2)
# The 2 values should be the same. They are :)


#part 4.7
K_est = estimate_intrinsics(Hs)
print("K_est:\n", K_est)
print("K_true:\n", K)
# The estimated K should be close to the true K. This one is very wrong :/ K_est is wrong





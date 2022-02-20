import sys
import numpy as np
import os
import scipy.misc
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    U, s, VT = np.linalg.svd(E)
    # compute R
    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    M = U.dot(Z).dot(U.T)
    Q1 = U.dot(W).dot(VT)
    R1 = np.linalg.det(Q1) * 1.0 * Q1

    Q2 = U.dot(W.T).dot(VT)
    R2 = np.linalg.det(Q2) * 1.0 * Q2

    # compute T
    T1 = U[:, 2].reshape(-1, 1)
    T2 = -U[:, 2].reshape(-1, 1)

    R_set = [R1, R2]
    T_set = [T1, T2]
    RT_set = []
    for i in range(len(R_set)):
        for j in range(len(T_set)):
            RT_set.append(np.hstack((R_set[i], T_set[j])))

    RT = np.zeros((4, 3, 4))
    for i in range(RT.shape[0]):
        RT[i, :, :] = RT_set[i]

    return RT

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    # construction of matrix A
    A1 = image_points[:,:1] * camera_matrices[:,2] - camera_matrices[:,0]
    A2 = image_points[:,1:] * camera_matrices[:,2] - camera_matrices[:,1]
    A = np.vstack((A1,A2))
    # svd to solve contraint minimizor P
    u,sig,vt = np.linalg.svd(A)
    P = vt[-1]
    P = P / P[-1]
    return P

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    # convert to homogeneous 3d coordinate
    if point_3d.shape[0]==3:
        point_3d = np.pad(point_3d,(0,1),constant_values=1)
    # projected 2d point in homogeneous coordinate
    homo_2d = camera_matrices @ point_3d
    # converge to 2d non-homogeneous coordinate
    point_2d = homo_2d[:,:2] / homo_2d[:,2:]
    # compute error
    error = point_2d - image_points
    return error.reshape(-1)

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    # number of cameras
    m = camera_matrices.shape[0]
    # convert to homogeneous 3d coordinate
    if point_3d.shape[0]==3:
        point_3d = np.pad(point_3d,(0,1),constant_values=1)
    # projected 2d point in homogeneous coordinate
    homo_2d = camera_matrices @ point_3d
    # vectorized version of Jacobian computation
    j1 = (camera_matrices[:,0,:3]*homo_2d[:,2:] - camera_matrices[:,2,:3]*homo_2d[:,:1]) / (homo_2d[:,2:]**2)
    j2 = (camera_matrices[:,1,:3]*homo_2d[:,2:] - camera_matrices[:,2,:3]*homo_2d[:,1:2]) / (homo_2d[:,2:]**2)
    J = np.ones((2*m,3))
    J[::2] = j1
    J[1::2] = j2
    return J

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # Linear estimate of 3d point as initial position
    P = linear_estimate_3d_point(image_points, camera_matrices)[:3]
    iterations = 10
    # Gauss-Newton optimization
    for i in range(iterations):
        e = reprojection_error(P, image_points, camera_matrices)
        J = jacobian(P, camera_matrices)
        P -= np.linalg.inv(J.T @ J) @ J.T @ e
    return P

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    # record number of points and number of cameras
    n,m = image_points.shape[:2]
    # obtain 4 possible R & T from Essential Matrix E
    RTs = estimate_initial_RT(E)
    # index of choice
    i_argmax = 0
    max_count = 0
    IO = np.pad(np.eye(3), ((0,0),(0,1)), constant_values=0)
    # deal with 4 combinations seperately
    for i in range(4):
        # obtain camera matrices from K & R & T combinations, shape (M,3,4)
        M1 = K @ IO
        R = RTs[i][:,:3]
        T = RTs[i][:,3]
        E2 = np.hstack((R.T, (-R.T@T).reshape(-1,1)))
        M2 = K @ E2
        camera_matrices = np.stack((M1,M2),axis=0)
        # record count
        count = 0
        for j in range(n):
            # nonlinear-estimate of 3d point shape (N,3)
            estimated_3d_point = nonlinear_estimate_3d_point(image_points[j], camera_matrices)
            # Projection onto 1st & 2nd camera coordinates
            c1 = estimated_3d_point
            c2 = E2@np.hstack((estimated_3d_point,1))
            # count the number of positive z-axis value for each R & T combinations
            count += int(c1[-1]>0) + int(c2[-1]>0)
        # update on the maximum count & argmax index values recursively
        if count>=max_count:
            i_argmax = i
            max_count = count
    return RTs[i_argmax]



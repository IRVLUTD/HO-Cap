"""
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de


About this file:
================
This file defines a wrapper for the loading functions of the MANO model.

Modules included:
- load_model:
  loads the MANO model from a given file location (i.e. a .pkl file location),
  or a dictionary object.

"""


import chumpy as ch
import numpy as np
import cv2


class Rodrigues(ch.Ch):
    dterms = "rt"

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


# def lrotmin(p):
#     if isinstance(p, np.ndarray):
#         p = p.ravel()[3:]
#         return np.concatenate(
#             [
#                 (cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel()
#                 for pp in p.reshape((-1, 3))
#             ]
#         ).ravel()
#     if p.ndim != 2 or p.shape[1] != 3:
#         p = p.reshape((-1, 3))
#     p = p[1:]
#     return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel() for pp in p]).ravel()


def lrotmin(pose):
    pose_cube = pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    R = rodrigues(pose_cube)  # (n_J, 3, 3), 1st is a global
    I_cube = np.broadcast_to(
        np.expand_dims(np.eye(3), axis=0), (R.shape[0] - 1, 3, 3)
    )  # (n_J, ([[1,0], [0,1]]))
    lrotmin = (R[1:] - I_cube).ravel()  # ((n_J-1)*3*3, )
    return lrotmin


def posemap(s):
    if s == "lrotmin":
        return lrotmin
    else:
        raise Exception("Unknown posemapping: %s" % (str(s),))


def rodrigues(r):
    """
    formula(1) in SMPL paper
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.
    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].
    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].
    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(theta.dtype).tiny)
    r_hat = r / theta
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    z_stick = np.zeros(theta.shape[0])
    # m is the skew symmetric of r_hat
    m = np.dstack(
        [
            z_stick,
            -r_hat[:, 0, 2],
            r_hat[:, 0, 1],
            r_hat[:, 0, 2],
            z_stick,
            -r_hat[:, 0, 0],
            -r_hat[:, 0, 1],
            r_hat[:, 0, 0],
            z_stick,
        ]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(np.expand_dims(np.eye(3), axis=0), [theta.shape[0], 3, 3])
    R = i_cube + (1 - cosTheta) * np.matmul(m, m) + sinTheta * m
    return R

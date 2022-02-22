import numpy as np

class Camera:

    def __init__(self, K, P, true_silhouette):
        pass

    def __init__(self, frame):
        self.image = frame[0]
        self.P = frame[1]
        self.K = frame[2]
        self.R = frame[3]
        self.T = frame[4][:,0]
        self.silhouette = frame[6]

    # Get the unit vector for the direction of the camera
    def get_camera_direction(self):
        # this is supposed to correspond to image coordinate of the image center
        # x: image coordinate of the image center
        x = np.array([self.image.shape[1] / 2,
             self.image.shape[0] / 2,
             1])
        # this solves for the X where K@X = x, projection of image center in camera coordinate, X is a 3-dimensional vector, a 'directional vector' in 3D cooridnate
        X = np.linalg.solve(self.K, x)
        # R.T @ K^{-1} @ x = (K@R)^{-1} @ x = X
        X = self.R.transpose().dot(X)
        # return a normalized direction of (K @ R)^{-1} @ x
        return X / np.linalg.norm(X)

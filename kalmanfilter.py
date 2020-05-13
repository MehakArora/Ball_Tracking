# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:23:48 2020

@author: Mehak
"""

class KalmanFilter(object):

    def __init__((bx,by), framerate):

        # delta T - time difference
        self.dt = 5*framerate

        # Previous state vector {x,y,vx,vy}
        self.xkp = np.array([bx, by, 0, 0])
        self.xk = self.xkp

        # Measured vector
        self.xm = np.array([0, 0, 0, 0])

        #State Transition Matrix
        self.F = np.eye(self.xkp.shape[0])
        self.F[0][2] = self.dt
        self.F[1][3] = self.dt

        # Initial Process Covariance Matrix
        self.Pkp = np.eye(self.xkp.shape[0])

        # Process Noise Covariance Matrix
        self.Qk = 10*np.eye(self.xkp.shape[0])

        # Control Matrix
        self.Bk = np.array([(self.dt**2)/2, self.dt])

        #Control Vector - initialised to acceleration due to gravity
        self.uk = np.array([-9.8])

        #Sensor Matrix
        self.Hk = np.array([[1,0,0,0],[0,1,0,0]])

        #Measurement covariance matrix
        self.R = 10*np.eye(self.xm.shape[0])

    def predict():

        #Predicted Vector
        self.xk = self.F @ self.xkp + self.Bk @ self.uk

        #Setting Previously predicted to current for next frame
        self.xkp = self.xk

        #Getting the updated process cov matrix
        self.Pkp = self.F @ self.Pkp @ self.F.T

        return self.xk

    def update((bx,by)):

        #Update Measurement Vector
        self.xm = np.array([bx,by])
        #Kalman Gain
        K = self.Pkp @ self.Hk.T @ np.linalg.inv(self.Hk @ self.Pkp @ self.Hk.T + self.R)

        #Most Likely state Vector
        self.xk = self.xk + K @ (self.xm - self.Hk @ self.xk)

        #Updated Process covariance Matrix
        self.Pkp = self.Pkp - K @ self.Hk @ self.Pkp

        #Setting previously predicted to current most likely for next frame
        self.xkp = self.xk

        return self.xk

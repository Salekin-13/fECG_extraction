import numpy as np

class RLS(object):
    """For initialization, if no priori knowledge is known: 
        P0=cI, where c is a large number
        X0 = random value"""
    
    def __init__(self, P0, x0, R):
        self.P0 = P0
        self.x0 = x0
        self.R = R #measurement noise covariance matrix

        self.currentTimestep_n = 0

        self.x_estimates = []
        self.x_estimates.append(x0) #this is what is being estimated by the filter

        #gain matrices
        self.gain_mats = [] #redundant

        #estimation error x-x'_k covariance matrix
        self.P_estimates = []
        self.P_estimates.append(P0)

        #estimated error matrices
        self.errors = [] #redundant



    def weight_update(self, measurement_y, C):
        #from eq 47: L(n) = R(n) + C(n)P(n-1)CT(n)
        #C is probably the weight matrix

        Lmatrix = self.R + np.matmul(C, np.matmul(self.P_estimates[self.currentTimestep_n], C.T))
        Lmatrix_inv = np.linalg.inv(Lmatrix)

        #gain matrix calculation from eqn 47: K(n) = P(n-1)CT(n) L-1(n)
        gain_K = np.matmul(self.P_estimates[self.currentTimestep_n], np.matmul(C.T, Lmatrix_inv)) #include lambda later
        self.gain_mats.append(gain_K)

        #calculate the error: e(n) = y(n) - x^(n)
        error = measurement_y - np.matmul(C, self.x_estimates[self.currentTimestep_n])
        self.errors.append(error)

        x_update = self.x_estimates[self.currentTimestep_n] + np.matmul(gain_K, error)
        self.x_estimates.append(x_update)

        #updating error covariance matrix from eq 50: P(n) = [I - K(n)C(n)]P(n-1)
        P_update = np.matmul(np.eye(np.size(self.x0), np.size(self.x0)) - np.matmul(gain_K, C), self.P_estimates[self.currentTimestep_n])
        self.P_estimates.append(P_update)

        self.currentTimestep_n = self.currentTimestep_n + 1






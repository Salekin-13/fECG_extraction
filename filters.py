import numpy as np

class RLS(object):
    """For initialization, if no priori knowledge is known:
        P(0) = delta*I 
        delta is initialized as a small +ve constant in dsp slide (gamma = delta_inv 1e9)
        weight initialization, w(0) = 0 (dsp slide)"""
    
    def __init__(self, delta, w0):
        self.delta = delta
        self.w0 = w0 #initialized with L
        #w0 is a 1D vector w/ L elements
        #initialized as a Lx1 col vector


        #self.R = R #measurement noise covariance matrix
        #disregard R for now

        self.currentTimestep_n = 0

        self.w_estimates = []
        self.w_estimates.append(w0) #this is what is being estimated by the filter: the weights


        #estimation error x-x'_k covariance matrix
        self.P_estimates = []
        P0 = delta*np.eye(np.size(w0)) #an LxL matrix
        self.P_estimates.append(P0)




    def weight_update(self, measurement_d, u, lamda):
        #from eq 47: L(n) = R(n) + C(n)P(n-1)CT(n)
        #C is probably the input matrix

        #the input sig u is assumed to be a nxm matrix
        #the weight is a column vector nx1
        #and P is nxn 

        Lmatrix = lamda*np.eye(np.size(u, axis=1)) + np.matmul(u.T, np.matmul(self.P_estimates[self.currentTimestep_n], u)) #the I matrix is 1x1; u is 32x1
        #Lmatrix is 1x1
        Lmatrix_inv = np.linalg.inv(Lmatrix)

        #gain matrix calculation from eqn 47: K(n) = P(n-1)CT(n) L-1(n)
        gain_K = np.matmul(self.P_estimates[self.currentTimestep_n], np.matmul(u, Lmatrix_inv)) #include lambda later, shape 32x1

        #calculate the estimation error: alpha(n) = d(n) - wT(n-1)u(n)
        error = measurement_d - np.matmul(u.T, self.w_estimates[self.currentTimestep_n]) #uTw is 1x1; so d needs to be a single element

        w_update = self.w_estimates[self.currentTimestep_n] + np.matmul(gain_K, error) #this looks like weight update
        #w(n) = w(n-1) + k(n)a(n)
        self.w_estimates.append(w_update)

        #updating error covariance matrix from eq 50: P(n) = [I - K(n)C(n)]P(n-1)
        P_update = self.P_estimates[self.currentTimestep_n] - np.matmul(gain_K, np.matmul(u.T, self.P_estimates[self.currentTimestep_n])) #32x32 update
        self.P_estimates.append((1/lamda)*P_update)

        self.currentTimestep_n = self.currentTimestep_n + 1






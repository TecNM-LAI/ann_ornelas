import numpy as np



class neuron(object):
    def __init__(self, x_in, w_in):
        self.v = x_in
        self.w = w_in
        self.z_in = 0
        self.z = 0
    def calc_ponderada(self):
        self.z_in = np.dot(self.v, self.w)
    def calc_output(self):
        self.z = 1 / (1 + np.exp(-self.z_in))
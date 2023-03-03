import numpy as np



neuron_pos = {"input": 0, "hidden": 1, "output": 2}


class neuron(object):
    def __init__(self):
        self.v = None
        self.w = None
        self.z_in = None
        self.z = None
        self.pos = None
    def build(self, x_in, w_in, pos):
        self.v = x_in
        self.w = w_in
        self.z_in = 0
        self.z = 0
        self.pos = pos
    def calc_output(self):
        if(self.pos != "input"):
            self.z_in = np.dot(self.v, self.w)
            self.z = 1 / (1 + np.exp(-self.z_in))
        else:
            self.z_in = self.v
            self.z = self.z_in

class layer(object):
    def __init__(self):
        self.neurons = None
        self.pos = None
    def build(self, neurons, pos):
        self.neurons = neurons
        self.pos = pos
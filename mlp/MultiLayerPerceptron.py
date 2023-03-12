import numpy as np



def sigmoid_act(x, derivate = False):
    sigmoid = lambda X: (1 / (1 + np.exp(-X)))
    if(derivate == True):
        f = sigmoid(x) * (1 - sigmoid(x))
    else: 
        f = sigmoid(x)
    return f


class neural_neuron(object):
    def __init__(self, w_in=np.array([]), layer="input", pos=1):
        self.v = np.array([])
        self.w = w_in
        self.z_in = 0
        self.z = 0
        self.layer = layer
        self.pos = pos
    def feedforward(self, x_in=np.array([])):
        self.v = x_in
        if(self.layer != "input"):
            self.z_in = np.dot(self.v, self.w)
            self.z = sigmoid_act(x=self.z_in, derivate=False)
        else:
            self.z_in = self.v[self.pos]
            self.z = self.z_in

class neural_layer(object):
    def __init__(self, w_in=np.array([[],[]]), bias=1, n_neurons=1, pos="input"):
        self.neurons = []
        self.v = np.array([])
        self.w = w_in
        self.pos = pos
        for i in range(n_neurons):
            if(self.pos != "input"):
                self.neurons.append(neural_neuron(w_in=self.w[:,i], layer=self.pos, pos=i+1))
            else:
                self.neurons.append(neural_neuron(layer=pos, pos=i+1))
        self.bias = bias
        self.y = np.array([])
        self.n_neurons = n_neurons
    def feedforward(self, x_in=np.array([])):
        if(self.pos != "input"):
            self.v = x_in
        else:
            self.v = np.insert(x_in, 0, self.bias)
        for neuron in range(self.n_neurons):
            self.neurons[neuron].feedforward(x_in=self.v)
            self.y = np.append(self.y, self.neurons[neuron].z)
        if(self.pos != "output"):
            self.y = np.insert(self.y, 0, self.bias)

class neural_network(object):
    def __init__(self, l_rate=0.3, epoch=1000, bias=1, 
                 layer_input_neurons=1, 
                 layer_hidden_neurons=1, 
                 layer_output_neurons=1):
        self.lin = layer_input_neurons
        self.lhn = layer_hidden_neurons
        self.lon = layer_output_neurons
        self.w_hidden = np.random.uniform(-1, 1, size=(self.lin+1, self.lhn))
        self.w_output = np.random.uniform(-1, 1, size=(self.lhn+1, self.lon))
        self.layer_input = neural_layer(bias=bias, n_neurons=self.lin, pos="input")
        self.layer_hidden = neural_layer(w_in=self.w_hidden, bias=bias, n_neurons=self.lhn, pos="hidden")
        self.layer_output = neural_layer(w_in=self.w_output, n_neurons=self.lon, pos="output")
        self.input_v = np.array([])
    def feedforward(self, x_in=np.array([])):
        self.input_v = x_in
        self.layer_input.feedforward(x_in=self.input_v)
        self.layer_hidden.feedforward(self.layer_input.y)
        self.layer_output.feedforward(self.layer_hidden.y)
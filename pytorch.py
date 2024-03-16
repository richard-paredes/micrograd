from random import random

from value import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self, x):
        cumulative = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        out = cumulative.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params
    

class MultiLayerPerceptron:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

n = MultiLayerPerceptron(3, [4,4,1]) # 3 neuron inputs, with 2 layers having 4 neurons and 1 output layer with 1 neuron

x_ins = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets (i.e. for input = [2.0, 3.0, -1.0] ->  desired_output = 1.0)
ypred = [n(x) for x in x_ins]

# implement mean squared error loss
losses = [(yout - ygt)**2 for ygt, yout in zip(ys, ypred)]
loss = sum((yout - ygt)**2 for ygt,yout in zip(ys, ypred))
loss.backward()

for p in n.parameters():
    p.data += -0.01 * p.grad

for k in range(20): # num epochs
    # forward pass
    ypred = [n(x) for x in x_ins]
    loss = sum((yout -ygt)**2 for ygt,yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters(): # zero-out the grad, since they will accumulate each iteration
        p.grad = 0.0         # and we want the backward pass to accumulate the loss derivatives
    loss.backward()

    # update/gradient descent
    for p in n.parameters():
        p.data += 0.01 * p.grad

    print(k, loss.data)
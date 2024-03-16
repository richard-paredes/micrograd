import math


class Value:

    def __init__(self, data, _children=[], _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    # equivalent to toString
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # operator override
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other], '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other], '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward
        return out
    
    def ___rmul(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    
    def __floordiv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data // other.data, [self, other], '//')
        return out
    
    def __mod__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data % other.data, [self, other], '%')
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * other.grad
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.data
        out._backward = _backward

        return out
    
    def tanh(self):
        other = other if isinstance(other, Value) else Value(other)
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self._grad = (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topological_graph = []
        visited = set()
        def build_topological_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topological_graph(child)
                topological_graph.append(v)
            
        build_topological_graph(self)
        self.grad = 1.0
        for node in reversed(self):
            node._backward()


    
# the leaf nodes are the inputs/weights that affect the loss function
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b;
e.label = 'e'

d = e+c
d.label = 'd'
f = Value(-2.0, label='f')
L = d*f; L.label = 'L'

print(d, d._prev)


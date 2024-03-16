class Value:

    def __init__(self, data, _children=[], _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0

    # equivalent to toString
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # operator override
    def __add__(self, other):
        out = Value(self.data + other.data, [self, other], '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, [self, other], '*')
        return out
    
    def __sub__(self, other):
        out = Value(self.data - other.data, [self, other], '-')
        return out
    
    def __truediv__(self, other):
        out = Value(self.data / other.data, [self, other], '/')
        return out
    
    def __floordiv__(self, other):
        out = Value(self.data // other.data, [self, other], '//')
        return out
    
    def __mod__(self, other):
        out = Value(self.data % other.data, [self, other], '%')
        return out

    def __pow__(self, other):
        out = Value(self.data ** other.data, [self, other], '**')
        return out
    
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

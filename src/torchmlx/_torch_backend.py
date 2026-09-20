import torch as _native


Tensor = _native.Tensor
tensor = _native.tensor
from_numpy = _native.from_numpy
arange = _native.arange
randint = _native.randint
chunk = _native.chunk
transpose = _native.transpose
zeros = _native.zeros
ones = _native.ones
zeros_like = _native.zeros_like
ones_like = _native.ones_like
cat = _native.cat
softmax = _native.softmax
triu = _native.triu
tril = _native.tril
topk = _native.topk
unique = _native.unique
exp = _native.exp
sin = _native.sin
cos = _native.cos
tanh = _native.tanh
sqrt = _native.sqrt
matmul = _native.matmul
outer = _native.outer
pow = _native.pow
polar = _native.polar
float16 = _native.float16
float32 = _native.float32
float64 = _native.float64
bfloat16 = _native.bfloat16
int8 = _native.int8
int16 = _native.int16
int32 = _native.int32
int64 = _native.int64
uint8 = _native.uint8
bool = _native.bool
half = float16
float = float32
double = float64
int = int32
long = int64
device = _native.device
backends = _native.backends
cuda = _native.cuda
pi = _native.pi


def categorical(logits, dim=-1, num_samples=1):
    probabilities = _native.softmax(logits, dim=dim)
    return _native.multinomial(probabilities, num_samples=num_samples)


def fallback(name):
    return getattr(_native, name)

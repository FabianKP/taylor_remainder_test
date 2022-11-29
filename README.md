# Taylor remainder test

This is a Python module that implements the Taylor remainder test.

## Usage example

```python

from math import cos, sin
import numpy as np
from taylor_remainder import taylor_remainder_test, taylor_remainder_test_jacobian


def fun1(x):
    return x[0] ** 2 - x[1] ** 3


def grad1(x):
    return np.array([2 * x[0], -3 * x[1] ** 2])

x = np.array([1., 2.])
taylor_remainder_test(fun1, grad1, x)
# True

def fun2(x):
    return x[0] ** 2 + x[1] ** 4


def grad2(x):
    return np.array([2 * x[0] ** 2, 4.001 * x[1] ** 3])

x = np.zeros(2)
taylor_remainder_test(fun2, grad2, x, sigma=0.1, eps=1e-10)
# False

def fun3(x):
    return np.array([x[0] ** 2 - x[1], cos(x[1])])


def jac(x):
    return np.array([[2 * x[0], -1.], [0., -sin(x[1])]])


x = np.zeros(2)
taylor_remainder_test_jacobian(fun=fun3, jac=jac, x=x, sigma=1.)
# True
```
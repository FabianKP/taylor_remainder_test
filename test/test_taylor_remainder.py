
from math import cos, sin
import numpy as np

from taylor_remainder import taylor_remainder_test, taylor_remainder_test_jacobian
from taylor_remainder.taylor_remainder import _single_taylor_test


def test_taylor_remainder_simple():
    def fun(x):
        return x[0] ** 2 - x[1] ** 3

    def grad(x):
        return np.array([2 * x[0], -3 * x[1] ** 2])
    x = np.array([1., 2.])
    assert taylor_remainder_test(fun, grad, x)


def test_taylor_remainder_highdim():
    dim = 1000
    def fun(x):
        return 0.5 * np.sum(np.square(np.exp(- 2 * x)))

    def grad(x):
        return - 2 * np.exp(-2 * x) * np.exp(-2 * x)

    x = np.zeros(dim)
    assert taylor_remainder_test(fun, grad, x, sigma=0.1, eps=1e-10)


def test_taylor_remainder_wrong():
    def fun(x):
        return x[0] ** 2 + x[1] ** 4

    def wrong_grad(x):
        return np.array([2 * x[0] ** 2, 4.001 * x[1] ** 3])

    x = np.zeros(2)
    passed = taylor_remainder_test(fun, wrong_grad, x, sigma=0.1, eps=1e-10)
    assert not passed


def test_taylor_remainder_jacobian():
    def fun(x):
        return np.array([x[0] ** 2 - x[1], cos(x[1])])

    def jac(x):
        return np.array([[2 * x[0], -1.], [0., -sin(x[1])]])

    x = np.zeros(2)
    assert taylor_remainder_test_jacobian(fun=fun, jac=jac, x=x)


def test_taylor_remainder_large_fun():
    s = 10
    dim = 1000
    def inner_fun(x):
        return x * np.power((1 - np.square(x)), s)

    def inner_jac(x):
        return np.power((1 - np.square(x)), s) - 2 * s * np.square(x) * np.power((1 - np.square(x)), s - 1)
    def fun(x):
        return np.sum(np.square(inner_fun(x)))

    def grad(x):
        j = inner_jac(x)
        g = j * fun(x)
        return g

    def wrong_grad(x):
        j = inner_jac(x)
        g = j * fun(x) + np.ones_like(x)
        return g

    x = np.zeros(dim)
    grad_passed = taylor_remainder_test(fun=fun, grad=grad, x=x)
    wrong_grad_passed = taylor_remainder_test(fun=fun, grad=wrong_grad, x=x)
    assert grad_passed and not wrong_grad_passed









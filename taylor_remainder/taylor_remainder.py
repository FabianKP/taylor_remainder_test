
import numpy as np


def taylor_remainder_test(fun: callable, grad: callable, x: callable, reduction: float = 2., num_tests: int = 100,
                          rtol: float = 1e-2, eps: float = 1e-10, pass_ratio: float = 1, sigma: float = 0.,
                          seed: int = None, return_count: bool = False):
    """
    Tests the gradient for a univariable function in a small neighbourhood of a point
    :math:`x` using the Taylor remainder test. The test is based on the fact that
    :math:`||f(x + h dx) - f(x) - h \\nabla f(x)^\\top dx|| = O(h^2)`.

    See also
    https://en.wikipedia.org/wiki/Taylor%27s_theorem#Taylor's_theorem_for_multivariate_functions.

    Parameters
    ---
    fun
        The function :math:`f`, must take a vector of shape (n,) as input and return a vector of shape (m, ). It is
        advised to scale `fun` such that for typical input, its input is on the scale O(1).
    grad
        The supposed gradient :math:`\\nabla f`. Must take input of shape (n,) and return vectors of shape (m, ).
    x
        The point at which the gradient is tested.
    num_tests
        Number of random points near x where Jacobian is tested.
    reduction
        Factor by which h is reduced in every step. Must be larger than 1 (default is 2).
    rtol
        Relative tolerance with which :math:`||f(x + h dx) - f(x) - h \\nabla f(x)^\\top dx|| = O(h^2)` has to hold.
    eps
        Desired numerical accuracy of the gradient. This means that the Taylor remainder should scale like
        O(h^2) for all h larger than `eps`.
    pass_ratio
        The ratio of single tests that need to pass (sometimes it is a good idea to have a pass ratio less than 1).
    sigma
        Noise with standard deviation `sigma` is added to `x` before each test. Defaults to 0, which means that
        every test is performed at `x`.
    seed
        Seed for `numpy.random`. Can be used to make the test deterministic.
    return_count
        If `True`, returns the number of tests that passed.

    Returns
    ---
    passed : bool
        True if at least `pass_ratio * 100%` of all tests passed.
    tests_passed : int
        The number of tests that passed. Only returned if `return_count` is set to `True`.
    """
    if seed is not None:
        np.random.seed(seed)
    if reduction <= 1.:
        raise ValueError("'reduction must be float larger than 1.")
    tests_passed = 0
    for i in range(num_tests):
        perturbation = sigma * np.random.randn(x.size)
        x_perturbed = x + perturbation
        dx = np.random.randn(x.size)
        if _single_taylor_test(fun=fun, grad=grad, x=x_perturbed, dx=dx, eps=eps, rtol=rtol, reduction=reduction):
            tests_passed += 1
    passed = (tests_passed >= pass_ratio * num_tests)
    if return_count:
        return passed, tests_passed
    else:
        return passed


def taylor_remainder_test_jacobian(fun: callable, jac: callable, x: callable, reduction: float = 2.,
                                   num_tests: int = 100, rtol: float = 1e-2, eps: float = 1e-10,
                                   pass_ratio: float = 0.95, sigma: float = 0., seed: int = 42,
                                   return_count: bool = False):
    """
    Tests the validity of the Jacobian for a multivariable function :math:`F` in a small neighbourhood of a point
    :math:`x` using the Taylor remainder test.
    What is actually tested is the gradient of :math:`f(x) = 0.5*||F(x)||^2`
    which is given by :math:`J(x).T F(x)`.

    Parameters
    ---
    fun :
        The function :math:`F`, must take a vector of shape (n,) as input and return a vector of shape (m, ). It is
        advised to scale `fun` such that `np.sum(np.square(fun(x))` is on the scale O(1) for expected `x`.
    jac :
        The supposed Jacobian :math:`J=DF`. Must return a numpy array of shape (m, n).
    x :
        The point at which the Jacobian is tested.
    num_tests :
        Number of random points near x where Jacobian is tested.
    reduction:
        Factor by which h is reduced in every step. Must be larger than 1 (default is 2).
    rtol :
        Relative tolerance with which :math:`||f(x + h dx) - f(x) - h \\nabla f(x)^\\top dx|| = O(h^2)` has to hold.
    eps :
        Desired numerical accuracy of the gradient.
    pass_ratio :
        The ratio of single tests that need to pass (sometimes it is a good idea to have a pass ratio less than 1).
    sigma :
        Noise with standard deviation `sigma` is added to `x` before each test. Defaults to 0, which means that
        every test is performed at `x`.
    seed :
        Seed for `numpy.random`. Can be used to make the test deterministic.
    return_count
        If `True`, returns the number of tests that passed.

    Returns
    ---
    passed : bool
        True if at least `ratio*100%` of all tests passed.
    tests_passed : int
        The number of tests that passed. Only returned if `return_count` is set to `True`.
    """
    def sum_squares_fun(v):
        return 0.5 * np.sum(np.square(fun(v)))

    def sum_squares_grad(v):
        return jac(v).T @ fun(v)
    return taylor_remainder_test(fun=sum_squares_fun, grad=sum_squares_grad, x=x, num_tests=num_tests,
                                 reduction=reduction, rtol=rtol, pass_ratio=pass_ratio, eps=eps, sigma=sigma, seed=seed,
                                 return_count=return_count)


def _single_taylor_test(fun: callable, grad: callable, x: np.array, dx: np.array, reduction: float, rtol: float,
                        eps: float) -> bool:
    """
    Performs a single Taylor remainder tests.
    """
    h = 1.
    r_ref = _second_order_remainder(fun=fun, grad=grad, x=x, dx=dx, h=h)
    r_prev = r_ref
    passed = True
    while r_prev > eps * (reduction ** 2):
        h = h / reduction
        r = _second_order_remainder(fun=fun, grad=grad, x=x, dx=dx, h=h)
        # last check wins
        thresh = (1 + rtol) * r_prev / (reduction ** 2)
        if r <= thresh:
            passed = True
        else:
            passed = False
        r_prev = r
    return passed


def _second_order_remainder(fun: callable, grad: callable, x: np.array, dx: np.array, h: float) -> float:
    """
    Computes second order taylor remainder.

    .. math::
        r = ||f(x+h dx) - f(x) - h grad(x) dx||_2.
    """
    r = np.linalg.norm(fun(x + h * dx) - fun(x) - h * grad(x) @ dx)
    return r

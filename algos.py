"""
The algorithms for the animations
"""
import warnings

import sympy as sp
import numpy as np

from typing_extensions import deprecated

import debug as dbg

TOLERANCE = 1e-9


@deprecated("Use Function.deriv()")
def deriv_(z, f, df=1e-5):
    """
    returns the derivative of f at z
    :param z: point on the complex plane
    :type z: complex
    :param f: the function to differentiate
    :type f: function or lambda
    :param df: the step size
    :type df: float
    :return: the derivative
    :rtype: dtype
    """
    df_direction_i = (f(z + df * 1j) - f(z)) / df
    df_direction_r = (f(z + df) - f(z)) / df
    return df_direction_r + 1j * df_direction_i


class Function:
    """
    A class for a function
    """

    def __init__(self, expression, variable='z'):
        """
        initializes the function
        :param expression: the expression of the function
        :type expression: str
        :param variable: the variable of the function
        :type variable: str
        """
        self.expression = expression
        self.variable = variable
        self.symbol = sp.symbols(variable)
        self.function = sp.sympify(expression)
        self.function_function = sp.lambdify(self.symbol, self.function)
        self.derivative = sp.diff(sp.sympify(expression), self.symbol)
        self.derivative_function = sp.lambdify(self.symbol, self.derivative)
        self.roots_ = sp.solve(self.derivative, self.symbol)
        self.roots = np.array([complex(root) for root in self.roots_])
        dbg.debug_print(f"roots: {repr(self.roots)}")
        dbg.debug_print(*self.roots, sep="\n")

    def __call__(self, z):
        """
        evaluates the function at z
        :param z: the point to evaluate the function at
        :type z: complex
        :return: the value of the function at z
        :rtype: dtype
        """
        return self.function_function(z)

    def deriv(self, z):
        """
        evaluates the derivative of the function at z
        :param z: the point to evaluate the derivative at
        :type z: complex
        :return: the value of the derivative at z
        """
        return self.derivative_function(z)


def newton_step_single(z, f):
    """
    returns the next point in the newton iteration
    :param z: starting point
    :type z: np.array
    :param f: function to find the roots
    :type f: Function
    :return:
    :rtype:
    """
    skip = np.ones_like(z)
    for root in f.roots:
        skip *= np.abs(z - root) < TOLERANCE
    _deriv = f.deriv(z)
    skip *= np.abs(_deriv) < TOLERANCE
    f_z = f(z)
    # skip
    _deriv += skip
    f_z *= 1 - skip
    res = z - f_z / _deriv
    return res

"""
The algorithms for the animations
"""
import numpy as np
import sympy as sp
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


def deriv_newton_step(z, f):
    """
    returns the next deriv in the newton iteration for debugging
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
    # skip
    _deriv += skip
    return _deriv


def func_newton_step(z, f):
    """
    returns the next function value in the newton iteration for debugging
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
    f_z *= 1 - skip
    return f_z


def sympy_to_opencl(expr_, variable='z'):
    """
    Convert a SymPy expression to OpenCL code
    :param expr_: SymPy expression
    :param variable: variable name in the expression
    :return: OpenCL code as string
    """
    import sympy.printing.c as c_print

    # Create a printer that handles complex operations
    # Claude 3.7 Sonnet
    class OpenCLPrinter(c_print.C89CodePrinter):
        def _print_Pow(self, expr):
            base = expr.base
            exp = expr.exp

            # Case 1: Integer exponent (use repeated multiplication)
            if exp.is_integer and exp.is_number:
                exp_val = int(exp)

                # Handle special cases
                if exp_val == 0:
                    return "((CTYPE)(1.0, 0.0))"
                elif exp_val == 1:
                    return self._print(base)
                elif exp_val == 2:
                    return f"mul2({self._print(base)}, {self._print(base)})"
                elif exp_val > 0:
                    # For positive integer powers, use repeated multiplication
                    result = self._print(base)
                    for _ in range(exp_val - 1):
                        result = f"mul2({result}, {self._print(base)})"
                    return result
                else:
                    # For negative integer powers, compute 1/z^|n|
                    result = self._print(base)
                    for _ in range(abs(exp_val) - 1):
                        result = f"mul2({result}, {self._print(base)})"
                    return f"div2(((CTYPE)(1.0, 0.0)), {result})"

            # Case 2: Real number exponent (use cpow function)
            elif exp.is_real and exp.is_number:
                return f"cpow_real({self._print(base)}, {float(exp)})"

            # Case 3: Complex exponent (general case)
            else:
                return f"cpow({self._print(base)}, {self._print(exp)})"

        @deprecated("broken")
        def __print_mul_depr(self, expr):
            terms = list(expr.args)
            if len(terms) == 2:
                return f"mul2({self._print(terms[0])}, {self._print(terms[1])})"
            else:
                result = self._print(terms[0])
                for term in terms[1:]:
                    result = f"mul2({result}, {self._print(term)})"
                return result

        def _print_Mul(self, expr, **kwargs):
            terms = list(expr.args)

            # Separate numeric coefficients from complex terms
            numeric_coefficient = 1
            complex_terms = []

            for term in terms:
                if term.is_number:
                    numeric_coefficient *= float(term)
                else:
                    complex_terms.append(term)

            # If we have both a numeric coefficient and complex terms
            if numeric_coefficient != 1 and complex_terms:
                # Handle the numeric coefficient as a scalar multiplication
                if len(complex_terms) == 1:
                    term = complex_terms[0]
                    return (f"mul2((CTYPE)({numeric_coefficient}, 0.f), "
                            f"{self._print(term)})")
                else:
                    # First multiply the complex terms together
                    result = self._print(complex_terms[0])
                    for term in complex_terms[1:]:
                        result = f"mul2({result}, {self._print(term)})"
                    # Then multiply by the numeric coefficient
                    return (f"((CTYPE)({numeric_coefficient} * {result}.x,"
                            f"{numeric_coefficient} * {result}.y))")

            # If we only have a numeric coefficient
            if not complex_terms:
                return f"((CTYPE)({numeric_coefficient}, 0.0))"

            # If we only have complex terms (no numeric coefficient, or it's 1)
            if len(complex_terms) == 1:
                return self._print(complex_terms[0])
            else:
                result = self._print(complex_terms[0])
                for term in complex_terms[1:]:
                    result = f"mul2({result}, {self._print(term)})"
                return result

        def _print_Add(self, expr, **kwargs):
            terms = list(expr.args)
            result = []
            for term in terms:
                if term.is_number:
                    result.append(f"(CTYPE)({self._print(term)}, 0.0f)")
                else:
                    result.append(self._print(term))
            return " + ".join(result)

        def _print_Symbol(self, expr):
            if str(expr) == variable:
                return variable
            return f"complex({super()._print_Symbol(expr)})"

    printer = OpenCLPrinter()
    return printer.doprint(expr_)

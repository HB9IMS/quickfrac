"""
This module contains functions for plotting and displaying data.
removed from main.py to avoid circular imports
"""

import numpy as np
from typing_extensions import deprecated
import colorsys
import algos as alg
import pygame as pg


def unitprint(value, /, unit=None, power=None):
    """Format a number to engineering units and prefixes."""

    if unit is None:
        unit = " "
    original_value = value
    sign = " "
    log1000 = 0
    if value != 0:
        if value < 0:
            sign = "-"
            value *= -1
        log1000 = np.log10(value) // 3 // (power if power is not None else 1)
        value *= 1000 ** -log1000

    if abs(log1000) > 6:
        return f"{sign}{f'{abs(original_value):.3e}': >7}{unit}"

    prefix = {0: "", -1: "m", -2: "µ", -3: "n", -4: "p", -5: "f", -6: "a",
              1: "k", 2: "M", 3: "G", 4: "T", 5: "P", 6: "E"
              }[log1000]

    return f"{sign}{f'{value:.3f}': >7} {prefix or ' '}{unit}" \
           f"{(f'^{power}' if power is not None else '')}"


def _lerp(a, b, t):
    """linear interpolation"""
    return a + (b - a) * t


def mmx(t, a=0., b=255.):
    """returns the least extreme value"""
    return min(b, max(a, t))


def timed(f):
    """time a function"""
    import time
    def fx(*args, **kwargs):
        f"""
        timed function
        {f.__doc__}"""
        start = time.perf_counter_ns()
        res = f(*args, **kwargs)
        end = time.perf_counter_ns()
        print(f'{f.__name__} took {unitprint((end - start) / 1e9, "s")}')
        return res

    return fx


def complex_to_rgb(z, scale=1):
    """
    converts a complex number to rgb
    :param z: the complex number
    :type z: complex
    :param scale: the scale of the color
    :type scale: float
    :return: the rgb value
    :rtype: np.ndarray[np.uint8, ...]
    """
    angle = np.angle(z) % (2 * np.pi)  # color
    mag = np.abs(z) * scale  # brightness
    assert 0 <= mag, f"mag must be greater than 0, not {mag}"
    r, g, b = colorsys.hsv_to_rgb(angle / 2 / np.pi, 1, mag)
    return (
        np.uint8(mmx(r * 255, 0, 255)),
        np.uint8(mmx(g * 255, 0, 255)),
        np.uint8(mmx(b * 255, 0, 255)),
    )


def complex_to_rgb_unlimited(z, scale=1):
    """
    converts a complex number to rgb with more than uint8 range
    """
    angle = np.angle(z) % (2 * np.pi)  # color
    mag = np.abs(z) * scale  # brightness
    assert 0 <= mag, f"mag must be greater than 0, not {mag}"
    r, g, b = colorsys.hsv_to_rgb(angle / 2 / np.pi, 1, mag)
    return (r * 255), (g * 255), (b * 255)


def _convert_point_to_pos(pos, frame_values, screen_value):
    return _lerp(0, screen_value, (pos - frame_values[0]) / (frame_values[1] - frame_values[0]))


import matplotlib.colors as mpl_colors


class Fractal:
    """Graphical class for a fractal"""

    def __init__(self, width, height,
                 function_,
                 frame_points,
                 symbol="z", dtype=np.complex128, function__=None):
        """
        initializes the fractal
        :param width: visual width
        :type width: int
        :param height: visual height
        :type height: int
        :param function_: base function for the next iteration as string
        :type function_: str
        :param symbol: the variable to use in the function
        :type symbol: str
        :param dtype: the dtype of the pixels
        :param frame_points: the two corners of the frame
        :param function__: a function object if available
        :type function__: alg.Function
        :type frame_points: tuple[tuple[complex, complex], ...]
        """
        self.width = width
        self.height = height
        if function__ is None:
            self.function = alg.Function(function_, symbol)
        else:
            self.function = function__
        self._function = function_
        self._symbol = symbol
        self.dtype = dtype
        self.frame = frame_points
        self.pixels = np.array(
            [[
                _lerp(self.frame[0][0], self.frame[1][0], j / self.height)
                + _lerp(self.frame[0][1], self.frame[1][1], i / self.width) * 1j
                for j in range(height)] for i in range(width)],
            dtype=dtype
        )
        self.fvalues = np.zeros_like(self.pixels)
        self.derivs = np.zeros_like(self.pixels)
        self.rendered = np.ones((width, height, 3), dtype=np.uint8)
        self.rendered_full = np.ones((width, height, 3), dtype=np.uint32)

    @timed
    def iterate(self):
        """
        iterates the fractal
        """
        self.pixels[:] = alg.newton_step_single(
            self.pixels,
            self.function
        )

    @timed
    def render_new(self):
        """faster renderer"""
        h = np.clip((np.angle(self.pixels) % (2 * np.pi)) / 2 / np.pi, 0, 1)
        s = np.clip(np.ones_like(h), 0, 1)
        v = np.clip(np.abs(self.pixels), 0, 1)
        rgb = mpl_colors.hsv_to_rgb(np.dstack((h, s, v)))
        self.rendered = (rgb * 255).astype(np.uint8)

    @timed
    @deprecated("use render_new(); faster")
    def render(self):
        """
        renders the fractal to internal array
        """
        self.rendered = np.array(
            [[complex_to_rgb(j) for j in i]
             for i in self.pixels]
        )

    @timed
    def render_unlimited(self):
        """faster renderer"""
        h = (np.angle(self.pixels) % (2 * np.pi)) / 2 / np.pi
        s = np.ones_like(h)
        v = np.abs(self.pixels)
        s /= np.max(s)
        self.rendered_full = mpl_colors.hsv_to_rgb(np.dstack((h, s, v)))

    @timed
    @deprecated("use render_unlimited_new(); faster")
    def render_unlimited_old(self):
        """render to more than uint8"""
        self.rendered_full = np.array(
            [[complex_to_rgb_unlimited(j) for j in i]
             for i in self.pixels]
        )

    # @timed
    def draw(self, screen):
        """
        draws the fractal to the screen provided
        :param screen: the screen to draw to
        """
        pg.surfarray.blit_array(screen, self.rendered)

    @deprecated("use Fractal.draw()")
    def draw_legacy(self, screen):
        """deprecated"""
        for i in range(self.height):
            for j in range(self.width):
                screen.set_at((i, j), self.rendered[j, i])

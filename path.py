# path.py
# encoding: utf-8

"""
tool for rendering paths
"""
import numpy as np

RENDER_TYPE = "image"

if RENDER_TYPE == "image":
    from PIL import Image
elif RENDER_TYPE == "pygame":
    pass
else:
    raise ValueError("invalid render type")

import sys
import pyopencl as cl
import algos
import gpu
from alive_progress import alive_bar


def show(rendered):
    """shows the rendered fractals"""
    if RENDER_TYPE == "image":
        images = [
            Image.fromarray(x.transpose(1, 0, 2)) for x in rendered
        ]
        for i in images:
            i.show()


def _geometric_interpolation(a, b, t):
    """
    interpolate exponentially
    :param a: start
    :param b: end
    :param t: 0-1
    """
    return (b / a) ** t * a


def _generate_points_between(frames, step_size):
    start, stop = np.array(frames)
    cstart = (start[0] + start[1]) * 0.5
    cstop = (stop[0] + stop[1]) * 0.5
    res = []
    for i in range(step_size):
        res.append([])
        val = i / step_size
        for j in (0, 1):
            res[-1].append([])
            for k in (0, 1):
                res[-1][-1].append(_geometric_interpolation(start[j][k],
                                                            stop[j][k],
                                                            val
                                                            )
                                   )
    return res


def generate_more_points(path):
    """make more points from less"""
    for i in zip(path[:-1], path[1:]):
        for j in _generate_points_between(i, 10):
            yield j


def render(path, resolution=None, function=None):
    """renders each point from the path"""
    if resolution is None:
        resolution = (1280, 960)
    if function is None:
        function = "z ** 3 - 1"
    with open(path, "r") as f:
        path_strings = f.read().splitlines()
        path_points = [eval(i) for i in path_strings]
    function_ = algos.Function(function, "z")
    with open(f"kernels/compute.cl") as f:
        compute_program = cl.Program(
            gpu.GLOBAL_CONTEXT,
            f.read()
            .replace("$f$", algos.sympy_to_opencl(function_.function))
            .replace("$d$", algos.sympy_to_opencl(function_.derivative))
        ).build()
    print(repr(algos.sympy_to_opencl(function_.function)), repr(function_.function))
    print(repr(algos.sympy_to_opencl(function_.derivative)), repr(function_.derivative))

    with open(f"kernels/render.cl") as f:
        render_program = cl.Program(
            gpu.GLOBAL_CONTEXT, f.read()
        ).build()
    fractals = [
        gpu.GPUFractal(
            *resolution,
            function,
            frame_points=i, function__=function_,
            kernels=(compute_program, render_program)
        ) for i in generate_more_points(path_points)
    ]
    for i in fractals:
        i.iterate_ntimes(1000)

    with alive_bar(len(fractals)) as bar:
        for i in fractals:
            i.finish_iterate()
            i.render_new()
            bar()

    show([x.rendered for x in fractals])


if __name__ == "__main__":
    render("paths/path_.txt",
           function=sys.argv[1] if len(sys.argv) > 1 else (input("f: ") or None),
           resolution=eval(sys.argv[2] if len(sys.argv) > 2 else (input("res: ") or "None"))
           )

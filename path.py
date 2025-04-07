# path.py
# encoding: utf-8

"""
tool for rendering paths
"""

RENDER_TYPE = "image"

if RENDER_TYPE == "image":
    from PIL import Image
elif RENDER_TYPE == "pygame":
    pass
else:
    raise ValueError("invalid render type")

import gpu
import tqdm


def show(rendered):
    """shows the rendered fractals"""
    if RENDER_TYPE == "image":
        images = [
            Image.fromarray(x.transpose(1, 0, 2)) for x in rendered
        ]
        for i in images:
            i.show()


def render(path, resolution=None, function=None):
    """renders each point from the path"""
    if resolution is None:
        resolution = (1280, 960)
    if function is None:
        function = "z ** 3 - 1"
    with open(path, "r") as f:
        path_strings = f.read().splitlines()
        path_points = [eval(i) for i in path_strings]
    fractals = [
        gpu.GPUFractal(
            *resolution,
            function,
            frame_points=i
        ) for i in path_points
    ]
    for j in tqdm.tqdm(range(250)):
        for i in fractals:
            i.iterate()

    for i in fractals:
        i.render_new()

    show([x.rendered for x in fractals])


if __name__ == "__main__":
    render("paths/path.txt")

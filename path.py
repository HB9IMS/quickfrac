# path.py
# encoding: utf-8

"""
tool for rendering paths
"""
import sys

import numpy as np
import pyopencl as cl
from alive_progress import alive_bar

import algos
import gpu

STEPS = 90
NTIMES = 1000
FPS = 3

RENDER_TYPE = "np"


def _array_to_mp4(video_array: np.ndarray, output_path: str, fps: int = 30):
    """
    Convert a 4D numpy array to an MP4 video.

    Parameters:
    - video_array (np.ndarray): A 4D numpy array of shape (frames, height, width, channels), channels must be 3 (RGB).
    - output_path (str): Path to the output .mp4 file.
    - fps (int): Frames per second for the output video.
    """
    if video_array.ndim != 4:
        raise ValueError("Input array must be 4D with shape (frames, height, width, 3)")

    num_frames, height, width, channels = video_array.shape

    # Normalize data if needed
    if video_array.dtype != np.uint8:
        video_array = np.clip(video_array, 0, 255).astype(np.uint8)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with alive_bar(num_frames) as bar:
        for frame in video_array:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            bar()

    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")


def _show(rendered):
    """shows the rendered fractals"""
    if RENDER_TYPE == "image":
        images = [
            Image.fromarray(x.transpose(1, 0, 2)) for x in rendered
        ]
        for i in images:
            i.show()
    elif RENDER_TYPE == "video":
        _array_to_mp4(np.array(rendered), "output.mp4", FPS)
    elif RENDER_TYPE == "np":
        np.array(rendered).dump("output.npz")


def _geometric0_interpolation(a, b, t):
    """
    interpolate exponentially, converging towards 0
    :param a: start
    :param b: end
    :param t: 0-1
    """
    return (b / a) ** t * a


def _generate_points_between(frames, steps):
    start, stop = np.array(frames)
    length_x, length_y, *cstart, scstart = start
    _, _, *cstop, scstop = stop
    total_vec = np.array(cstop) - cstart
    scales = [_geometric0_interpolation(scstart, scstop, 1 - i / steps)
              for i in range(steps)]
    movement_scaling_factor = np.linalg.norm(total_vec) / sum(scales)
    centers = [cstart + total_vec * sum(scales[:i]) * movement_scaling_factor
               for i in range(steps)]
    res = [[[cy + length_x * scale, cx + length_y * scale],
            [cy - length_x * scale, cx - length_y * scale]] for ((cx, cy), scale) in zip(centers, scales)]
    return res


def generate_more_points(path, bar=None):
    """make more points from less"""
    for i in zip(path[:-1], path[1:]):
        for j in _generate_points_between(i, STEPS):
            yield j
            bar()


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

    with open(f"kernels/render.cl") as f:
        render_program = cl.Program(
            gpu.GLOBAL_CONTEXT, f.read()
        ).build()
    with alive_bar(STEPS * (len(path_points) - 1)) as bar:
        fractals = [
            gpu.GPUFractal(
                *resolution,
                function,
                frame_points=i, function__=function_,
                kernels=(compute_program, render_program)
            ) for i in generate_more_points(path_points, bar=bar)
        ]
    with alive_bar(len(fractals)) as bar:
        for i in fractals:
            i.iterate_ntimes(NTIMES)
            bar()

    with alive_bar(len(fractals)) as bar:
        for i in fractals:
            i.finish_iterate()
            i.render()
            bar()

    _show([x.rendered for x in fractals])


class PGVideoPlayer:
    def __init__(self, array):
        self.array = array
        self.fps = FPS
        self.frame = 0
        self.clock = pg.time.Clock()
        self.running = True
        self.screen = pg.display.set_mode(array[0].shape[:2])

    def run(self):
        pg.surfarray.blit_array(self.screen, self.array[self.frame])
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
        self.clock.tick(self.fps)


if __name__ == "__main__":
    if "-i" in sys.argv:
        sys.argv.remove("-i")
        RENDER_TYPE = "image"
    if "-v" in sys.argv:
        RENDER_TYPE = "video"
        sys.argv.remove("-v")
    if "-z" in sys.argv:
        RENDER_TYPE = "np"
        sys.argv.remove("-z")
    if "-p" in sys.argv:
        RENDER_TYPE = "pygame"
        sys.argv.remove("-p")

    if RENDER_TYPE == "image":
        from PIL import Image
    elif RENDER_TYPE == "pygame":
        import pygame as pg
    elif RENDER_TYPE == "video":
        import cv2
    elif RENDER_TYPE == "np":
        pass
    else:
        raise ValueError("invalid render type")

    render("paths/path_.txt",
           function=(sys.argv[1] if len(sys.argv) > 1
                     else None),
           resolution=eval(sys.argv[2] if len(sys.argv) > 2
                           else "None"
                           )
           )

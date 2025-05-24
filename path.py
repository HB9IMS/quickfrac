# path.py
# encoding: utf-8

"""
tool for rendering paths
"""

import numpy as np
import pyopencl as cl
from tqdm import tqdm

import algos
import debug
import gpu

STEPS = 90
NTIMES = 1000
FPS = 3

HOLD_AT_END = True
OUT_FILE_NAME = "outputs/output.mp4"
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

    video_array = video_array.transpose(0, 2, 1, 3)

    num_frames, height, width, channels = video_array.shape

    # Normalize data if needed
    if video_array.dtype != np.uint8:
        video_array = np.clip(video_array, 0, 255).astype(np.uint8)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in tqdm(video_array):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    if HOLD_AT_END:
        frame_bgr = cv2.cvtColor(video_array[-1], cv2.COLOR_RGB2BGR)
        for i in range(fps):
            out.write(frame_bgr)

    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")


def _show(rendered):
    """shows the rendered fractals"""
    if RENDER_TYPE == "image":
        images = [
            Image.fromarray(x_.transpose(1, 0, 2)) for x_ in rendered
        ]
        for i in images:
            i.show()
    elif RENDER_TYPE == "video":
        _array_to_mp4(np.array(rendered), OUT_FILE_NAME, FPS)
    elif RENDER_TYPE == "np":
        np.array(rendered).dump(OUT_FILE_NAME)


def _geometric0_interpolation(a, b, t):
    """
    interpolate exponentially, converging towards 0
    :param a: start
    :param b: end
    :param t: 0-1
    """
    return (b / a) ** t * a


def _generate_points_between(frames, steps):
    # fmt of input {dimensions[1] / 100 * scale}, {dimensions[0] / 100 * scale}, {cx}, {cy}, {scale}
    start, stop = np.array(frames)
    length_x, length_y, *cstart, scstart = start
    _, _, *cstop, scstop = stop
    total_vec = np.array(cstop) - cstart
    scales = [_geometric0_interpolation(scstart, scstop, i / steps)
              for i in range(steps)]
    movement_scaling_factor = 1 / sum(scales)
    centers = [cstart + total_vec * sum(scales[:i]) * movement_scaling_factor
               for i in range(steps)]
    res = [[[cy + length_x * scale / scstart, cx + length_y * scale / scstart],
            [cy - length_x * scale / scstart, cx - length_y * scale / scstart]] for ((cx, cy), scale) in zip(centers, scales)]
    if debug.DEBUG:
        print(f"scale: {scales}")
        print(f"centers: {centers}")
        print(f"movement_scaling_factor: {movement_scaling_factor}")
        print(f"res: {res}")
        print(f"exporting to ./DEBUG/path_{debug.tick_counter()}.npz")
        np.savez_compressed(f"./DEBUG/path_{debug.VARIABLES.c}.npz", res=res)
    return res


def generate_more_points(path, bar=None):
    """make more points from less"""
    for i in zip(path[:-1], path[1:]):
        for j in _generate_points_between(i, STEPS):
            yield j
            if bar is not None:
                bar()


def render(path, resolution=None, function=None, num_workers=None):
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
            ("#define DOUBLE\n" if gpu.DOUBLE else "")
            + f.read()
            .replace("$f$", algos.sympy_to_opencl(function_.function))
            .replace("$d$", algos.sympy_to_opencl(function_.derivative))
        ).build()

    with open(f"kernels/render.cl") as f:
        render_program = cl.Program(
            gpu.GLOBAL_CONTEXT,
            ("#define DOUBLE\n" if gpu.DOUBLE else "")
            + f.read()
        ).build()
    fractals = [
        gpu.GPUFractal(
            *resolution,
            function,
            frame_points=i, function__=function_,
            kernels=(compute_program, render_program)
        ) for i in tqdm(generate_more_points(path_points))
    ]
    for i in tqdm(fractals):
        i.iterate_ntimes(NTIMES)

    for i in tqdm(fractals):
        i.finish_iterate()
        i.render()

    _show([x_.rendered for x_ in fractals])


def render_sequential(path, resolution=None, function=None, num_workers=1):
    """renders each point from the path, one after the other (for memory)"""
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
            ("#define DOUBLE\n" if gpu.DOUBLE else "")
            + f.read()
            .replace("$f$", algos.sympy_to_opencl(function_.function))
            .replace("$d$", algos.sympy_to_opencl(function_.derivative))
        ).build()

    with open(f"kernels/render.cl") as f:
        render_program = cl.Program(
            gpu.GLOBAL_CONTEXT,
            ("#define DOUBLE\n" if gpu.DOUBLE else "")
            + f.read()
        ).build()

    fractals = [
        gpu.GPUFractal(
            *resolution,
            function, function__=function_,
            frame_points=[[1, 1], [0, 0]],
            kernels=(compute_program, render_program)
        ) for _ in range(num_workers)
    ]

    results = []
    points = list(generate_more_points(path_points))

    for i in tqdm(range(0, len(points), num_workers)):
        batch = points[i:i + num_workers]
        for fractal, point in zip(fractals, batch):
            fractal.frame = point
            fractal.reset()
            fractal.iterate_ntimes(NTIMES)
        for fractal in fractals:
            fractal.finish_iterate()
            fractal.render()
            results.append(fractal.rendered.copy())

    _show(results)


class PGVideoPlayer:
    """unused"""

    def __init__(self, array):
        self.array = array
        self.fps = FPS
        self.frame = 0
        self.clock = pg.time.Clock()
        self.running = True
        self.screen = pg.display.set_mode(array[0].shape[:2])

    def run(self):
        """unused"""
        pg.surfarray.blit_array(self.screen, self.array[self.frame])
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
        self.clock.tick(self.fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fractal renderer')
    x = parser.add_argument
    x('-i', '--image', action='store_const', const='image', dest='render_type', help='Render as image')
    x('-v', '--video', action='store_const', const='video', dest='render_type', help='Render as video')
    x('-z', '--numpy', action='store_const', const='np', dest='render_type', help='Render as numpy array')
    x('-p', '--pygame', action='store_const', const='pygame', dest='render_type', help='Render using pygame')
    x('--sequential', action='store_true', help='Use sequential rendering')
    x('-w', '--workers', type=int, default=1, dest='workers', help='Number of workers')
    x('-f', '--path', type=str, default='paths/path_.txt', dest='path', help='Path to path')
    x('-o', '--output', type=str, default='output', dest='output', help='Output file name')
    x('function', nargs='?', default=None, help='Function to render')
    x('resolution', nargs='?', type=eval, default=None, help='Resolution')
    x('fps', nargs='?', type=eval, help='Frames per second')
    x('steps', nargs='?', type=eval, help='Number of steps')
    x('ntimes', nargs='?', type=eval, help='Number of iterations')
    x('-d', '--debug', action='store_true', dest="debug", help='Manually activate debug mode')

    args = parser.parse_args()

    RENDER_TYPE = args.render_type
    renderer = render_sequential if args.sequential else render
    OUT_FILE_NAME = args.output

    if args.debug:
        debug.DEBUG = True
        debug.debug_print("Debug mode activated")

    if RENDER_TYPE == "image":
        from PIL import Image
    elif RENDER_TYPE == "pygame":
        import os

        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        import pygame as pg
    elif RENDER_TYPE == "video":
        import cv2
    elif RENDER_TYPE == "np":
        pass
    else:
        raise ValueError("invalid render type")

    if args.fps is not None:
        FPS = args.fps
    if args.steps is not None:
        STEPS = args.steps
    if args.ntimes is not None:
        NTIMES = args.ntimes

    renderer(args.path,
             function=args.function,
             resolution=args.resolution,
             num_workers=args.workers
             )

"""
Used to render images
"""

import numpy as np
from PIL import Image

from graphical import timed

import gpu


@timed
def render_image(frame, resolution, function, ntimes=100, tiles=(1, 1)):
    """
    render an image from a frame and a function
    :param frame: the frame points
    :type frame: tuple[tuple[float, float], tuple[float, float]]
    :param resolution: the resolution of the image
    :type resolution: tuple[int, int]
    :param function: the function to render
    :type function: str
    :param ntimes: the number of iterations
    :type ntimes: int
    :param tiles: number of tiles to split the render into (width_tiles, height_tiles)
    :type tiles: tuple[int, int]
    :return: the image
    :rtype: np.ndarray[np.uint8]
    """
    yes = np.linspace(frame[0][0], frame[1][0], resolution[1])
    xes = np.linspace(frame[0][1], frame[1][1], resolution[0]) * 1j
    starting_positions = (xes[None, :] + yes[:, None]).astype(np.complex128)
    finished_image = np.ones((resolution[0], resolution[1], 3), dtype=np.uint8)
    x_tiles, y_tiles = tiles
    arrays = [[starting_positions[i::x_tiles, j::y_tiles].T.copy() for j in range(y_tiles)] for i in range(x_tiles)]
    for i in range(x_tiles):
        for j in range(y_tiles):
            print(f"Tile [{i}, {j}]")
            fractal = gpu.GPUFractal(
                arrays[i][j].shape[0], arrays[i][j].shape[1],
                function, frame,
                array_in=arrays[i][j]
            )
            fractal.iterate_ntimes(ntimes)
            fractal.finish_iterate()
            fractal.render()
            finished_image[j::y_tiles, i::x_tiles] = fractal.rendered
    return finished_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fractal image renderer')
    x = parser.add_argument
    x('-f', '--function', type=str, default="UNSET", help='The function to render')
    x('-r', '--resolution', type=int, nargs=2, default=(1280, 960), help='The resolution of the image')
    x('-n', '--ntimes', type=int, default=100, help='The number of iterations')
    x('-o', '--output', type=str, default="outputs/output_{idx}.png",
      help='The output file with {idx} as the index'
      )
    x('-p', '--path', type=str, default="./paths/path_.txt", help='The path to the path file')
    x('-t', '--tiles', type=int, nargs=2, default=(1, 1),
      help='Number of tiles to split the render into (width_tiles height_tiles)'
      )

    args = parser.parse_args()

    if args.function == "UNSET":
        print("No function provided, please enter")
        args.function = input("> ") or "z ** 3 - 1"

    with open(args.path, "r") as f:
        path_strings = f.read().splitlines()
        path_points = [eval(i) for i in path_strings]

    # convert to frame points
    # fmt of input {dimensions[1] / 100 * scale}, {dimensions[0] / 100 * scale}, {cx}, {cy}, {scale}
    frame_points = [((i[3] + i[0], i[2] + i[1]), (i[3] - i[0], i[2] - i[1])) for i in path_points]

    images = [render_image(frame, args.resolution, args.function, args.ntimes, args.tiles) for frame in frame_points]
    for idx, image in enumerate(images):
        Image.fromarray(image.transpose(1, 0, 2)).save(args.output.format(idx=idx))

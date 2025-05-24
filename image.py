"""
Used to render images
"""

from PIL import Image

import gpu


def render_image(frame, resolution, function, ntimes=100):
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
    :return: the image
    :rtype: np.ndarray[np.uint8]
    """
    fractal = gpu.GPUFractal(resolution[0], resolution[1], function, frame)
    fractal.iterate_ntimes(ntimes)
    fractal.finish_iterate()
    fractal.render()
    return fractal.rendered


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fractal image renderer')
    x = parser.add_argument
    x('-f', '--function', type=str, default="UNSET", help='The function to render')
    x('-r', '--resolution', type=int, nargs=2, default=(1280, 960), help='The resolution of the image')
    x('-n', '--ntimes', type=int, default=100, help='The number of iterations')
    x('-o', '--output', type=str, default="outputs/output_{idx}.png", 
      help='The output file with {idx} as the index')
    x('-p', '--path', type=str, default="./paths/path_.txt", help='The path to the path file')

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

    images = [render_image(frame, args.resolution, args.function, args.ntimes) for frame in frame_points]
    for idx, image in enumerate(images):
        Image.fromarray(image.transpose(1, 0, 2)).save(args.output.format(idx=idx))

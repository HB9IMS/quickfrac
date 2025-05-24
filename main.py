"""
main document
"""

import sys

import matplotlib.pyplot as plt
from PIL import Image

import gpu
from graphical import *
from graphical import _convert_point_to_pos, lerp

function = "z ** 3 - 1"
GPU = True
GUI = True
ANTIALIAS = False
FULLSCREEN = True


def test_fractal(pathfile=sys.stdout):
    global function
    """test function"""
    import time, ctypes
    ctypes.windll.user32.SetProcessDPIAware()
    f_type = gpu.GPUFractal if GPU else Fractal
    if GUI:
        pg.init()
        if FULLSCREEN:
            size = (0, 0)
            screen = pg.display.set_mode(size, pg.FULLSCREEN)  # adapt for any screen
        else:
            k = 2
            size = (640 * k, 480 * k)
            del k
            screen = pg.display.set_mode(size)

        info = pg.display.Info()

        # Retrieve screen width and height
        dimensions = (info.current_w, info.current_h)
    else:
        dimensions = (640, 480)
        screen = None

    x1, y1 = dimensions[0] / 100, dimensions[1] / 100
    x2, y2 = -dimensions[0] / 100, -dimensions[1] / 100
    cx, cy = 0, 0
    scale = 1

    def update_xy():
        """update the frame coordinates"""
        nonlocal x1, x2, y1, y2, cx, cy, scale
        x, y = dimensions[1] / 100 * scale, dimensions[0] / 100 * scale
        x1, y1 = cy + x, cx + y
        x2, y2 = cy - x, cx - y

    update_xy()

    fractal = f_type(
        dimensions[0], dimensions[1],
        function,
        frame_points=((x1, y1), (x2, y2)),
        antialias=ANTIALIAS
    )

    def reset_fractal():
        """resets the fractal"""
        nonlocal fractal, cx, cy, scale
        cx, cy = 0, 0
        scale = 1
        update_xy()
        fractal = f_type(dimensions[0], dimensions[1],
                         function, frame_points=((x1, y1), (x2, y2)),
                         antialias=ANTIALIAS
                         )
        # if GPU:
        #     gpu.cl.enqueue_copy(fractal.queue, fractal.pixel_buffer, fractal.pixels)

    def reset_fractal_no_pos():
        """resets the fractal without changing position"""
        nonlocal fractal
        fractal.frame = ((x1, y1), (x2, y2))
        fractal.reset()
        fractal.render()
        # print(fractal.frame)

    this_time = 0
    if not GUI:
        fractal.iterate()
        fractal.render()
        Image.fromarray(
            fractal.rendered
        ).show()
        return
    fractal.render()
    do_iterate = False
    show_cross = True

    c = 0

    while True:
        this_time, prev_time = time.perf_counter_ns(), this_time
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return
            if event.type == pg.KEYDOWN:
                match event.key:
                    case pg.K_ESCAPE:
                        pg.quit()
                        return
                    case pg.K_i:
                        fractal.iterate()
                        fractal.render()
                    case pg.K_SPACE:
                        do_iterate = not do_iterate
                    case pg.K_d:
                        print(fractal.pixels)
                    case pg.K_b:
                        pass  # used for breakpoints
                    case pg.K_p:
                        fractal.render_unlimited()
                        plt.imshow(fractal.rendered_full / np.max(fractal.rendered_full))
                        plt.show()
                    case pg.K_s:
                        Image.fromarray(fractal.rendered.transpose(1, 0, 2)).show()
                    case pg.K_r:
                        reset_fractal()
                        do_iterate = False
                    case pg.K_RETURN:
                        fractal.render()
                    case pg.K_c:
                        if GPU:
                            fractal.render_cpu()
                    case pg.K_l:
                        if GPU:
                            fractal.iterate_cpu()
                            fractal.render()
                    case pg.K_m:
                        if GPU:
                            fractal.iterate_cpu()
                            fractal.render_cpu()
                    case pg.K_DOLLAR:
                        function = input(function)
                        reset_fractal()
                    case pg.K_DOWN:
                        cy -= 0.1 * scale
                        update_xy()
                    case pg.K_UP:
                        cy += 0.1 * scale
                        update_xy()
                    case pg.K_LEFT:
                        cx += 0.1 * scale
                        update_xy()
                    case pg.K_RIGHT:
                        cx -= 0.1 * scale
                        update_xy()
                    case pg.K_PAGEUP:
                        scale *= 2 ** .5
                        update_xy()
                    case pg.K_PAGEDOWN:
                        scale /= 2 ** .5
                        update_xy()
                    case pg.K_1:
                        scale /= 8
                        update_xy()
                    case pg.K_2:
                        scale *= 8
                        update_xy()
                    case pg.K_TAB:
                        reset_fractal_no_pos()
                        fractal.render()
                        c = 0
                    case pg.K_END:
                        print(f"{dimensions[1] / 100 * scale}, {dimensions[0] / 100 * scale}, "
                              f"{cx}, {cy}, {scale}",
                              file=pathfile
                              )
                        print(c)
                        c = 0
                    case pg.K_x:
                        show_cross = not show_cross
                # get mouse position on button press relative to fractal and move frame there
            if event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()
                pos_ = np.array(pos) / np.array(dimensions)
                frame = np.array(fractal.frame)
                pos__ = [*lerp(frame[0, ::-1], frame[1, ::-1], pos_)]
                cx, cy = pos__
                update_xy()
            if event.type == pg.MOUSEWHEEL:
                scale *= 2 ** (event.y * 0.1)
        # draw fractal
        if do_iterate:
            c += 1
            fractal.iterate()
            fractal.render()
        fractal.draw(screen)
        # draw fps
        frame = np.array(fractal.frame)
        xes = (x1, x2)
        ys = (y1, y2)
        _point_list = [(_convert_point_to_pos(_y, frame[:, 1], dimensions[0] - 1),
                        _convert_point_to_pos(_x, frame[:, 0], dimensions[1] - 1))
                       for _x in xes for _y in ys]
        for i in _point_list:
            for j in _point_list:
                if show_cross:
                    pg.draw.line(screen, "#ffffff", i, j)
        pg.display.set_caption(str(int(1e9 / (this_time - prev_time))) + ' fps')
        pg.display.flip()


if __name__ == '__main__':
    with open("paths/path_.txt", "w") as file:
        test_fractal(file)

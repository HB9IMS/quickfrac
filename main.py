"""
main document
"""

import sys

import matplotlib.pyplot as plt
from PIL import Image

import gpu
from graphical import *
from graphical import _convert_point_to_pos, _lerp

function = "z ** 3 - 1"
GPU = True
GUI = True
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
            size = (640, 480)
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
        x, y = dimensions[0] / 100 * scale, dimensions[1] / 100 * scale
        x1, y1 = cy + x, cx + y
        x2, y2 = cy - x, cx - y

    update_xy()

    fractal = f_type(
        dimensions[0], dimensions[1],
        function,
        frame_points=((x1, y1), (x2, y2))
    )

    def reset_fractal():
        """resets the fractal"""
        nonlocal fractal, cx, cy, scale
        cx, cy = 0, 0
        scale = 1
        update_xy()
        fractal = f_type(dimensions[0], dimensions[1], function, frame_points=((x1, y1), (x2, y2)))
        if GPU:
            gpu.cl.enqueue_copy(fractal.queue, fractal.pixel_buffer, fractal.pixels)

    def reset_fractal_no_pos():
        """resets the fractal without changing position"""
        nonlocal fractal
        fractal = f_type(dimensions[0], dimensions[1], function, frame_points=((x1, y1), (x2, y2)))

    this_time = 0
    if not GUI:
        fractal.iterate()
        fractal.render_new()
        Image.fromarray(
            fractal.rendered
        ).show()
        return
    fractal.render_new()
    do_iterate = False
    show_cross = True
    last_click_pos = None
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
                        fractal.render_new()
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
                        last_click_pos = None
                    case pg.K_RETURN:
                        fractal.render_new()
                    case pg.K_c:
                        if GPU:
                            fractal.render_cpu()
                    case pg.K_l:
                        if GPU:
                            fractal.iterate_cpu()
                            fractal.render_new()
                    case pg.K_m:
                        if GPU:
                            fractal.iterate_cpu()
                            fractal.render_cpu()
                    case pg.K_y:
                        if GPU:
                            fractal.show_differences()
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
                    case pg.K_TAB:
                        reset_fractal_no_pos()
                        fractal.render_new()
                    case pg.K_END:
                        print(f"{x1, y1}, {x2, y2}", file=pathfile)
                    case pg.K_x:
                        show_cross = not show_cross
                # get mouse position on button press relative to fractal
            if event.type == pg.MOUSEBUTTONDOWN and False:  # skip because broken_
                pos = pg.mouse.get_pos()
                pos_ = np.array(pos) / np.array(dimensions)
                pos__ = [_lerp(x1, x2, pos_[0]), _lerp(y1, y2, pos_[1])]
                print(pos__)
                fractal.rendered[pos[0]][pos[1]][0] ^= 0xff
                fractal.rendered[pos[0]][pos[1]][1] ^= 0xff
                fractal.rendered[pos[0]][pos[1]][2] ^= 0xff
                if last_click_pos is not None and np.any(pos__ != last_click_pos):
                    fractal = f_type(dimensions[0], dimensions[1], function,
                                     frame_points=(last_click_pos[::-1], pos__[::-1])
                                     )
                    print(f"new: {(last_click_pos[::-1], pos__[::-1])}")
                    do_iterate = False
                    last_click_pos = None
                else:
                    last_click_pos = pos__
        # draw fractal
        if do_iterate:
            fractal.iterate()
            fractal.render_new()
        fractal.draw(screen)
        # draw fps
        frame = np.array(fractal.frame)
        xes = (x1, x2)
        ys = (y1, y2)
        _point_list = [(_convert_point_to_pos(_y, frame[:, 1], dimensions[0]),
                        _convert_point_to_pos(_x, frame[:, 0], dimensions[1]))
                       for _x in xes for _y in ys]
        for i in _point_list:
            for j in _point_list:
                if show_cross:
                    pg.draw.line(screen, "#ffffff", i, j)
        pg.display.set_caption(str(int(1e9 / (this_time - prev_time))) + ' fps')
        pg.display.flip()


if __name__ == '__main__':
    with open("paths/path.txt", "w") as file:
        test_fractal(file)

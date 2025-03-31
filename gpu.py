"""
For the GPU thingies
"""

import numpy as np
import pyopencl as cl

import main


class GPUFractal(main):
    """
    A class for a fractal on the GPU
    """

    def __init__(self, width, height,
                 function_,
                 frame_points,
                 symbol="z", dtype=np.complex128,
                 kernel_path=None):
        """initializer"""
        if kernel_path is None:
            kernel_path = "kernels"
        super().__init__(width, height,
                         function_,
                         frame_points,
                         symbol, dtype
                         )

        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx, None)

        # buffers
        inout_flags = (cl.mem_flags.READ_WRITE
                       | cl.mem_flags.COPY_HOST_PTR)

        self.pixel_buffer = cl.Buffer(
            self.ctx,
            inout_flags,
            self.pixels.nbytes,
            hostbuf=self.pixels
        )
        self.rendered_buffer = cl.Buffer(
            self.ctx,
            inout_flags,
            self.rendered.nbytes,
            hostbuf=self.rendered
        )

        # programs
        self.compute_program = cl.Program(
            self.ctx, open(f"{kernel_path}/compute.cl").read()
        ).build()
        self.render_program = cl.Program(
            self.ctx, open(f"{kernel_path}/render.cl").read()
        ).build()

    def render_new(self):
        """faster renderer"""
        self.render_program.render(
            self.queue,
            self.rendered.shape,
            None,
            self.rendered_buffer,
            self.pixel_buffer,
        )

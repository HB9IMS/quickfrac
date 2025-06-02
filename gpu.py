"""
For the GPU thingies
"""

import numpy as np
import pyopencl as cl

import algos
import graphical as gp

GLOBAL_CONTEXT = cl.create_some_context()

dev = GLOBAL_CONTEXT.devices[0]
exts = dev.get_info(cl.device_info.EXTENSIONS).split()

DOUBLE = dev.get_info(cl.device_info.DOUBLE_FP_CONFIG) != 0

if not DOUBLE:
    @lambda x: x()
    def _():
        import sys
        print("WARNING: SINGLE PRECISION", file=sys.stderr)
        del sys

del dev, exts


class GPUFractal(gp.Fractal):
    """
    A class for a fractal on the GPU
    """

    def __init__(self, width, height,
                 function_,
                 frame_points,
                 symbol="z", dtype=None,
                 kernel_path=None, function__=None, kernels=None,
                 antialias=False, array_in=None):
        """initializer"""
        if dtype is None:
            dtype = np.complex128 if DOUBLE else np.complex64
            if not DOUBLE:
                @lambda x_: x_()
                def _():
                    import sys
                    print("WARNING: SINGLE PRECISION", file=sys.stderr)
                    del sys

        if kernel_path is None:
            kernel_path = "kernels"
        super().__init__(width, height,
                         function_,
                         frame_points,
                         symbol, dtype, function__
                         )
        self.antialias = antialias
        if self.antialias:
            x = np.linspace(self.frame[0][0], self.frame[1][0], self.height)
            y = np.linspace(self.frame[0][1], self.frame[1][1], self.width * 3) * 1j
            self.pixels = (x[None, :] + y[:, None]).astype(self.dtype)

        if array_in is not None:
            self.pixels = array_in
        self.ctx = GLOBAL_CONTEXT
        self.queue = cl.CommandQueue(self.ctx, None)
        self.kernel_path = kernel_path

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
        if kernels is None:
            with open(f"{kernel_path}/compute.cl") as f:
                self.compute_program = cl.Program(
                    self.ctx,
                    ("#define DOUBLE\n" if DOUBLE else "")
                    + f.read()
                    .replace("$f$", algos.sympy_to_opencl(self.function.function))
                    .replace("$d$", algos.sympy_to_opencl(self.function.derivative))
                ).build()
            print(repr(algos.sympy_to_opencl(self.function.function)), repr(self.function.function))
            print(repr(algos.sympy_to_opencl(self.function.derivative)), repr(self.function.derivative))

            with open(f"{kernel_path}/render.cl") as f:
                self.render_program = cl.Program(
                    self.ctx,
                    ("#define DOUBLE\n" if DOUBLE else "")
                    + f.read()
                ).build()
        else:
            self.compute_program, self.render_program = kernels

    def reload_compute_kernel(self):
        """
        reloads the compute kernel
        useful when loading new functions
        """
        with open(f"{self.kernel_path}/compute.cl") as f:
            self.compute_program = cl.Program(
                self.ctx,
                ("#define DOUBLE\n" if DOUBLE else "")
                + f.read()
                .replace("$f$", algos.sympy_to_opencl(self.function.function))
                .replace("$d$", algos.sympy_to_opencl(self.function.derivative))
            ).build()

    def render(self):
        """faster renderer"""
        if self.antialias:
            self.render_program.render_subpixel_aa(
                self.queue,
                self.rendered.shape,
                None,
                self.rendered_buffer,
                self.pixel_buffer,
                np.int32(self.pixels.shape[1])
            )
        else:
            self.render_program.render(
                self.queue,
                self.rendered.shape,
                None,
                self.rendered_buffer,
                self.pixel_buffer,
                np.int32(self.pixels.shape[1])
            )
        cl.enqueue_copy(
            self.queue, self.rendered,
            self.rendered_buffer
        )
        self.queue.finish()

    def render_cpu(self):
        """renders using cpu (legacy/debug)"""
        super().render()

    def iterate_cpu(self):
        """iterates using cpu (legacy/debug)"""
        # First, copy data from GPU to CPU
        cl.enqueue_copy(self.queue, self.pixels, self.pixel_buffer)
        self.queue.finish()

        # Perform CPU iteration
        super().iterate()

        # Copy updated data back to GPU
        cl.enqueue_copy(self.queue, self.pixel_buffer, self.pixels)
        self.queue.finish()

    def iterate(self):
        """
        iterates the fractal
        """
        self.iterate_ntimes(1)
        self.finish_iterate()

    def iterate_ntimes(self, n):
        """
        iterates the fractal n times
        :param n: number of times to iterate
        """
        self.compute_program.step_n(
            self.queue,
            self.pixels.shape,
            None,
            self.pixel_buffer,
            np.int32(self.pixels.shape[1]),
            np.int32(n)
        )

    def finish_iterate(self):
        """finish iterating (only needed with ntimes)"""
        cl.enqueue_copy(
            self.queue,
            self.pixels,
            self.pixel_buffer
        )
        self.queue.finish()

    def reset(self):
        """resets the fractal"""
        super().reset()
        if self.antialias:
            x_vals = np.linspace(self.frame[0][0], self.frame[1][0], self.height * 3, dtype=np.float64)
            y_vals = np.linspace(self.frame[0][1], self.frame[1][1], self.width, dtype=np.float64)
            self.pixels = (np.array(np.meshgrid(x_vals, y_vals))
                           .transpose(2, 1, 0)
                           .reshape(-1)
                           .view(self.dtype)
                           .reshape(self.height * 3, self.width))
        cl.enqueue_copy(self.queue, self.pixel_buffer, self.pixels)
        self.queue.finish()

"""
For the GPU thingies
"""

import numpy as np
import pyopencl as cl

import algos
import graphical as gp

GLOBAL_CONTEXT = cl.create_some_context()


class GPUFractal(gp.Fractal):
    """
    A class for a fractal on the GPU
    """

    def __init__(self, width, height,
                 function_,
                 frame_points,
                 symbol="z", dtype=np.complex128,
                 kernel_path=None, function__=None, kernels=None,
                 antialias=False):
        """initializer"""
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
        self.roots_buffer = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            self.function.roots.nbytes,
            hostbuf=self.function.roots
        )

        # programs
        if kernels is None:
            with open(f"{kernel_path}/compute.cl") as f:
                self.compute_program = cl.Program(
                    self.ctx,
                    f.read()
                    .replace("$f$", algos.sympy_to_opencl(self.function.function))
                    .replace("$d$", algos.sympy_to_opencl(self.function.derivative))
                ).build()
            print(repr(algos.sympy_to_opencl(self.function.function)), repr(self.function.function))
            print(repr(algos.sympy_to_opencl(self.function.derivative)), repr(self.function.derivative))

            with open(f"{kernel_path}/render.cl") as f:
                self.render_program = cl.Program(
                    self.ctx, f.read()
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
                f.read()
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
        self.compute_program.step(
            self.queue,
            self.pixels.shape,
            None,
            self.pixel_buffer,
            np.int32(self.pixels.shape[1]),
            self.roots_buffer,
            np.int32(self.function.roots.shape[0]),
        )
        cl.enqueue_copy(
            self.queue,
            self.pixels,
            self.pixel_buffer
        )
        self.queue.finish()

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
            self.roots_buffer,
            np.int32(self.function.roots.shape[0]),
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
            x_values = np.linspace(self.frame[0][0], self.frame[1][0], self.height * 3, dtype=np.float64)
            y_values = np.linspace(self.frame[0][1], self.frame[1][1], self.width, dtype=np.float64)
            self.pixels = (np.array(np.meshgrid(x_values, y_values))
                           .transpose(2, 1, 0)
                           .reshape(-1)
                           .view(np.complex128)
                           .reshape(self.height * 3, self.width))
        cl.enqueue_copy(self.queue, self.pixel_buffer, self.pixels)
        self.queue.finish()

"""
For the GPU thingies
"""

import numpy as np
import pyopencl as cl

import matplotlib.colors as mpl_colors

import algos
import main


class GPUFractal(main.Fractal):
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
        self.kernel_path = kernel_path

        self.gpu_fvalues = np.zeros_like(self.pixels)
        self.gpu_derivs = np.zeros_like(self.pixels)

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
        self.gpu_fvalues_buffer = cl.Buffer(
            self.ctx,
            inout_flags,
            self.gpu_fvalues.nbytes,
            hostbuf=self.gpu_fvalues
        )
        self.gpu_derivs_buffer = cl.Buffer(
            self.ctx,
            inout_flags,
            self.gpu_derivs.nbytes,
            hostbuf=self.gpu_derivs
        )

        # programs
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

    @main.timed
    def render_new(self):
        """faster renderer"""
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
        super().render_new()

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

    @main.timed
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
            self.gpu_fvalues_buffer,
            self.gpu_derivs_buffer
        )
        cl.enqueue_copy(
            self.queue,
            self.pixels,
            self.pixel_buffer
        )
        cl.enqueue_copy(
            self.queue,
            self.gpu_derivs,
            self.gpu_derivs_buffer,
        )
        cl.enqueue_copy(
            self.queue,
            self.gpu_fvalues,
            self.gpu_fvalues_buffer
        )
        self.queue.finish()

    def show_differences(self):
        """compares the cpu and gpu results"""
        pixels_before = self.pixels.copy()
        self.iterate()
        self.pixels = pixels_before
        self.iterate_cpu()

        print(np.allclose(self.gpu_fvalues, self.fvalues))
        print(np.allclose(self.gpu_derivs, self.derivs))
        display = self.gpu_fvalues - self.fvalues
        h = np.clip((np.angle(display) % (2 * np.pi)) / 2 / np.pi, 0, 1)
        s = np.clip(np.ones_like(h), 0, 1)
        v = np.clip(np.abs(display), 0, 1)
        rgb = mpl_colors.hsv_to_rgb(np.dstack((h, s, v)))
        self.rendered = (rgb * 255).astype(np.uint8)

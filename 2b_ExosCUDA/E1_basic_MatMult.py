"""
Code adapted from  https://shephexd.github.io/development/2017/02/19/pycuda.html

If using google colab:
* Click on Runtime (excecution) and select Change runtime type (modifier le type d'excecution).
  Then select GPU in Hardware Acceleration (accélérateur matériel)
* Start your session by installing pycuda with the command:
  -> !pip install pycuda
"""


import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import time

# -- initialize the device
import pycuda.autoinit


#get device information
MyDevice=pycuda.driver.Device(0)
MyDevice.get_attributes()

#define the kernel
kernel_code_template = """
__global__ void MatrixMulKernel(float *a, float *b, float *c)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N,
    //   to produce one element of P.
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
        float Aelement = a[ty * %(MATRIX_SIZE)s + k];
        float Belement = b[k * %(MATRIX_SIZE)s + tx];
        Pvalue += Aelement * Belement;
    }

    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
"""

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads
# -> use MyDevice.get_attributes() to get this information
MATRIX_SIZE = 32

# create two random square matrices
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# compute reference on the CPU to verify GPU computation
time_start=time.time()
c_cpu = np.dot(a_cpu, b_cpu)
time_end=time.time()
print('enlapsed time (CPU):',time_end-time_start,' seconds')


# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# get the kernel code from the template
# by specifying the constant MATRIX_SIZE
kernel_code = kernel_code_template % {
    'MATRIX_SIZE': MATRIX_SIZE
    }

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")

# call the kernel on the card
time_start=time.time()

matrixmul(
    # inputs
    a_gpu, b_gpu,
    # output
    c_gpu,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (MATRIX_SIZE, MATRIX_SIZE, 1),
    )

time_end=time.time()
print('enlapsed time (GPU):',time_end-time_start,' seconds')


# print the results
print("-" * 80)
print("Matrix A (GPU):")
print(a_gpu.get())

print("-" * 80)
print("Matrix B (GPU):")
print(b_gpu.get())

print("-" * 80)
print("Matrix C (GPU):")
print(c_gpu.get())

print("-" * 80)
print("CPU-GPU difference:")
print(c_cpu - c_gpu.get())




#QUESTION 1: Comprenez bien chaque partie du code

#QUESTION 2: Comparez le temps necessaire pour la multiplication en CPU et en GPU pour MATRIX_SIZE = 8, 16, 32. Qu'en pensez vous?

#QUESTION 3: Par quelle methode simple pourriez vous rendre la parallelisation GPU competitive par rapport a la methode CPU (i.e. avec numpy)?

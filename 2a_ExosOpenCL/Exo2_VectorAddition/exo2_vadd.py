"""
Vadd: Element wise addition of two vectors (c = a + b)

History: C version written by Tim Mattson, December 2009
         C version Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
         Ported to Python by Tom Deakin, July 2013
         2017-2020: Adapted by Laurent Risser for practical courses at ISAE

If using google colab:
* Click on Runtime (excecution) and select Change runtime type (modifier le type d'excecution).
  Then select GPU in Hardware Acceleration (accélérateur matériel)
* Start your session by installing pycuda with the command:
  -> !pip install pyopencl
"""



import pyopencl as cl
import numpy
from time import time
import sys



#------------------------------------------------------------------------------
# 1) OpenCL Kernel: vadd
#
# To compute the elementwise sum c = a + b
#
# Input: a and b float vectors of length count
# Output c float vector of length count holding the sum a + b
#------------------------------------------------------------------------------

kernelsource = """
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsigned int count)
{
    int i = get_global_id(0);
    if (i < count)
        c[i] = a[i] + b[i];
}
"""

#------------------------------------------------------------------------------
# 2) Main procedure
#------------------------------------------------------------------------------

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#2.1) OpenCL initialization
#
#-> QUESTION : A quoi sert 'queue' ainsi que la commande cl.Program(...)
#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create a compute context
# Ask the user to select a platform/device on the CLI
context = cl.create_some_context()

#REMARQUE IMPORTANTE :
#VOUS POUVEZ DE MANIERE ALTERNATIVE UTILISER LE MEILLEUR CHOIX ISSU DE
#L'EXERCICE 1 POUR CREER LE CONTEXTE AVEC LES COMMANDES (EN ADAPTANT
#LES CHOIX DANS LES LISTES):
#
#platforms = cl.get_platforms()
#print('Possible platforms: '+str(platforms))
#devices = platforms[0].get_devices()
#print('Possible devices: '+str(devices))
#context = cl.Context([devices[1]])

# Create a command queue
queue = cl.CommandQueue(context)

# Create the compute program from the source buffer and build it
program = cl.Program(context, kernelsource).build()


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#2.2) generate the data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# length of vectors a, b and c
LENGTH = 1024



# Create a and b vectors and fill with random float values
h_a = numpy.random.rand(LENGTH).astype(numpy.float32)
h_b = numpy.random.rand(LENGTH).astype(numpy.float32)
# Create an empty c vector (a+b) to be returned from the compute device
h_c = numpy.empty(LENGTH).astype(numpy.float32)


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#2.3) allocate memory in the device and copy information
#
#-> QUESTION : Quelle est la difference entre (h_a,h_b,h_c) et (d_a,d_b,d_c)
#
#-> QUESTION : Quelle est la difference principale entre la creation
#              des buffers d_a, d_b et celle de d_c
#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create the input (a, b) arrays in device memory and copy data from host
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
# Create the output (c) array in device memory
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#2.4) run the OpenCL kernel and copy the result from the compute device
#to the standard memory
#
#-> QUESTION : Pourquoi utilise-t-on "set_scalar_arg_dtypes" ?
#
#-> QUESTION : Que signifient les différents paramètres de "vadd(queue,...)" ?
#
#-> QUESTION : Que fait-on avec "cl.enqueue_copy(...)" ?
#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Start the timer
rtime = time()

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
vadd = program.vadd
vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])
vadd(queue, h_a.shape, None, d_a, d_b, d_c, LENGTH)

# Wait for the commands to finish before reading back
queue.finish()
rtime = time() - rtime
print("The kernel ran in", rtime, "seconds")

cl.enqueue_copy(queue, h_c, d_c)



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#2.5) test the result
#
#-> QUESTION : Si vous pouvez utiliser plusieurs devices, testez lequel est le
#              plus rapide.
#
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

TOL = 0.001  # tolerance used in floating point comparisons


correct = 0;
for a, b, c in zip(h_a, h_b, h_c):
    # assign element i of a+b to tmp
    tmp = a + b
    # compute the deviation of expected and output result
    tmp -= c
    # correct if square deviation is less than tolerance squared
    if tmp*tmp < TOL*TOL:
        correct += 1
    else:
        print("tmp", tmp, "h_a", a, "h_b", b, "h_c", c)

# Summarize results
print("C = A+B:", correct, "out of", LENGTH, "results were correct.")

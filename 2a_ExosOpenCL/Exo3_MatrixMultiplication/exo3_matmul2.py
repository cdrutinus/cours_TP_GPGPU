
"""
Matrix Multiplication Driver

This is a driver program to test various ways of computing the product:
                    C = A * B

A and B are constant matrices, square and the order is set as a constant, ORDER.
This is so we can make a quick test of the multiplication result.


History: C++ version written by Tim Mattson, August 2010
         Modified by Simon McIntosh-Smith, September 2011
         Modified by Tom Deakin and Simon McIntosh-Smith, October 2012
         Ported to Python by Tom Deakin, July 2013
         2017-2020 Adapted by Laurent Risser for practical courses at ISAE

If using google colab:
* Click on Runtime (excecution) and select Change runtime type (modifier le type
  d'excecution).
  Then select GPU in Hardware Acceleration (accélérateur matériel)
* Start your session by installing pycuda with the command:
  -> !pip install pyopencl
* The kernels that can be found in the .cl files should be entered as the
  "kernelsource=..." command in exo2_vadd.py.
* At some point, you will have to  copy/paste two functions of helper.py into
  your code
"""




import pyopencl as cl
import numpy
from time import time
import helper


#--------------------------------------------------------------------------------
#PART 1) DEFINITIONS (TO MANUALLY CHANGE IF YOU WANT)
#--------------------------------------------------------------------------------
# Order of the square matrices A, B and C
ORDER = 1024

# A elemetns are constant and equal to AVAL
AVAL = 3.0

# B elemetns are constant and equal to BVAL
BVAL = 5.0

# tolerance used in floating point comparisons
TOL = 0.001

# Max dim for NDRange
DIM = 2

# number of times to do each multiplication
COUNT = 1


#--------------------------------------------------------------------------------
#PART 2) INIT
#--------------------------------------------------------------------------------

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#2.1) Data generation
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# A[N][N], B[N][N], C[N][N]
N = ORDER;


# Number of elements in the matrix
size = N * N


# A matrix
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)

# B matrix
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)

# C matrix
h_C = numpy.empty(size).astype(numpy.float32)


#REMARQUE IMPORTANTE : LES MATRICES h_A, h_B, h_C SONT STOQUEES DANS DES VECTEURS DE TAILLE N*N

#QUESTION : AVANT DE VOUS LANCER DANS LA MULTIPLICATION DE MATRICES SOUS OPENCL, VOUS
#POUVEZ TESTER LA MULTIPLICATION EN PUR PYTHON AVEC LA FONCTION "seq_mat_mul_sdot"
#DE "helper.py"... NOTEZ BIEN LES TEMPS DE CALCUL !
#POUR UNE COMPARAISON PLUS JUSTE ENTRE PYTHON ET OPENCL, LA DERNIERE BOUCLE DE CETTE
#FONCTION ie "for k in range(N)" POURRAIT ETRE TRAITEE AVEC NUMPY.
#

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#2.2) initiate opencl
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Set up OpenCL    ->   on my ubuntu PC: PYOPENCL_CTX='1:0'
context = cl.create_some_context()

#ALTERNATIVE RECOMMANDEE POUR LA CREATION DE CONTEXTE (EN ADAPTANT LES
#CHOIX DANS LES LISTES):
#
#platforms = cl.get_platforms()
#devices = platforms[0].get_devices()
#context = cl.Context([devices[1]])



queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

#--------------------------------------------------------------------------------
# PART 3) OpenCL matrix multiplication
#
#QUESTION : PRENEZ LE TEMPS DE BIEN COMPRENDRE LES DIFFERENCES ENTRE LES DIFFERENTES
#STRATEGIES ET NOTEZ LES DIFFERENTS TEMPS DE CALCULS. SI VOUS AVEZ PLUSIEURS CHOIX
#DE DEVICES SOUS LA MAIN, IL EST AUSSI INTERESSANTS DE COMPARER L'EFFICACITE DES
#DIFFERENTES STRATEGIES EN FONCTION DES PROPRIETES DU DEVICE.
#
#REMARQUE : IL SE PEUT QUE LES FONCTIONS LES PLUS AVANCEES NE MARCHENT PAS SI LE
#DEVICE N'EST PAS UNE CARTE GRAPHIQUE (TYPIQUEMENT UN CPU).
#--------------------------------------------------------------------------------

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# STRATEGY 3.a)  Naive (equivalent to matmul1.py)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#initiate the computations
kernelsource = open("./CL/C_elem.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])
print("\n===== OpenCL, matrix mult, C(i,j) per work item, order", N, "======\n")

h_C.fill(0.0)


# Do the multiplication
start_time = time()

mmul(queue, (N, N), None, N, d_a, d_b, d_c)
queue.finish()

run_time = time() - start_time

cl.enqueue_copy(queue, h_C, d_c)

# show results
helper.results(N, h_C, run_time,AVAL,BVAL,TOL)

#--------------------------------------------------------------------------------
# STRATEGY  3.b)   C row per work item
#--------------------------------------------------------------------------------

#initiate the computations

kernelsource = open("./CL/C_row.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])

print("\n===== OpenCL, matrix mult, C row per work item, order", N, "======\n")

h_C.fill(0.0)


# Do the multiplication

start_time = time()

mmul(queue, (N,), (int(ORDER/16),), N, d_a, d_b, d_c)
queue.finish()

run_time = time() - start_time

cl.enqueue_copy(queue, h_C, d_c)

# show results
helper.results(N, h_C, run_time,AVAL,BVAL,TOL)

#--------------------------------------------------------------------------------
# STRATEGY 3.c)    C row per work item, A row in pivate memory
#--------------------------------------------------------------------------------

#initiate the computations

kernelsource = open("./CL/C_row_priv.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None])

print("\n===== OpenCL, matrix mult, C row, A row in priv mem, order", N, "======\n")

h_C.fill(0.0)


# Do the multiplication

start_time = time()

mmul(queue, (N,), (int(ORDER/16),), N, d_a, d_b, d_c)
queue.finish()

run_time = time() - start_time

cl.enqueue_copy(queue, h_C, d_c)

# show results
helper.results(N, h_C, run_time,AVAL,BVAL,TOL)

#--------------------------------------------------------------------------------
#  STRATEGY 3.d)    C row per work item, A row pivate, B col local
#--------------------------------------------------------------------------------

#initiate the computations

kernelsource = open("./CL/C_row_priv_bloc.cl").read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, None, None, None, None])

print("\n===== OpenCL, mat mult, C row, priv A, B cols loc, order", N, "======\n")

h_C.fill(0.0)


# Do the multiplication
start_time = time()

localmem = cl.LocalMemory(numpy.dtype(numpy.float32).itemsize * N)
mmul(queue, (N,), (int(ORDER/16),), N, d_a, d_b, d_c, localmem)
queue.finish()

run_time = time() - start_time

cl.enqueue_copy(queue, h_C, d_c)

# show results
helper.results(N, h_C, run_time,AVAL,BVAL,TOL)

#--------------------------------------------------------------------------------
#  STRATEGY 3.e)    blocked
#--------------------------------------------------------------------------------


#DERNIERE QUESTION (TRES TRES OPTIONELLE) :  IMPLEMENTER LA SOLUTION DE TUILAGE
#(TILING) VUE EN COURS. BON COURAGE ;-)

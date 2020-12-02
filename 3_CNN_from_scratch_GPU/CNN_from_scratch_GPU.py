
"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#EXERCICE : ACCELERATION GPU D'UN CNN (basique) CODE EN PYTHON/NUMPY
#
# -> Le but de cet exercice est
#    (1) de bien comprendre comment les parametres d'un reseau de neurones sont appris
#     et en particulier comment fonctionne la backpropagation. L'implementation  python
#    de differentes couches de CNN est donnee. Le CNN apprend alors a reconnaitre des 0
#    et des 1 du jeu de donnees MNIST.
#    (2) rendre le code le plus rapide possible en utilisant de l'acceleration GPU avec
#    openCL ou bien CUDA.
#
#-> Une fois le code compris ... un seul objectif : l'accelerer avec du code GPU !
#
#-> Conseil : Si le temps est limite, focalisez vous sur l'amélioration de
#             'conv_backward', avec au debut un code openCL ou CUDA qui recode
#             l'algorithme sequentiel puis ensuite sa parallelisation. Il sera alors
#             evident de mesurer l'impact de la parallelisation !
#             Une alternative est de se focaliser sur 'dense_forward' qui permet
#             de comparer du code bien optimise avec numpy contre du code GPU. Cette
#             alternative est plus simple a coder mais les gains potentiels sont
#             moindres.
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""



import numpy as np
import matplotlib.pyplot as plt
from time import time


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                     convolutional layer
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def init_convolution_filter(size):
  """
  size must be a list of the shape [size_h,size_w]
  """
  w=(0.5+np.random.randn(size[0],size[1]))/(size[0]*size[1])
  return w


def conv_forward(input, w):
    """
    INSPIRED BY: https://gist.github.com/neodelphis
    Remark: here stride=1 / no padding / image has only one layer / only one filter

    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N observations, each of height H and
    width W.
    We convolve each input with a filter of height HH and width WW.
    Input:
    - input: Input data of shape (N, H, W)
    - w: Filter weights of shape (HH, WW)

    Returns a tuple of:
    - output: Output data, of shape (N, H', W') where H' and W' are given by H'=H and W'=W
    - cache: (input, w)
    """

    N, H, W = input.shape
    HH, WW = w.shape

    # dimensions de la sortie (tests sur la validité des choix necessaires ensuite / sinon padding)
    H_ = H
    W_ = W

    output = np.zeros((N, H_, W_))

    # Version sans vectorisation
    for n in range(N):       # On parcourt toutes les images
            for i in range(H_): # indices du résultat
                for j in range(W_):
                    for k in range(HH): # indices du filtre
                        for l in range(WW):
                          if i+k<H and j+l<W:   #test whether we're inside the image
                                output[n,i,j] += input[n, i+k, j+l] * w[k, l]

    cache = (input, w)
    return output, cache



def conv_backward(grad_output, cache):
    """
    INSPIRED BY: https://gist.github.com/neodelphis
    Remark: here stride=1 / no padding / image has only one layer / only one filter

    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - grad_output: Upstream derivatives. -> Gradient of the loss at the output of the layer
    - cache: A tuple of (input, w) as in conv_forward_naive

    Returns a tuple of:
    - grad_input: Gradient of the loss at the input of the layer wrt input
    - grad_w: Gradient of the loss at the input of the layer wrt w
    """

    # Récupération des variables
    input, w = cache

    # Initialisations
    grad_input = np.zeros_like(input)
    grad_w = np.zeros_like(w)

    # Dimensions
    N, H, W = input.shape
    HH, WW = w.shape
    _, H_, W_ = grad_output.shape   #H_ and W_ sould be equal to H and W

    # Version sans vectorisation
    for n in range(N):       # On parcourt toutes les images
            for i in range(HH): # indices du résultat
                for j in range(WW):
                    for k in range(H_): # indices du filtre
                        for l in range(W_):
                          if i+k<H_ and j+l<W_:
                                grad_w[i,j] += input[n, i+k, j+l] * grad_output[n, k, l]


    # Version sans vectorisation
    for n in range(N):       # On parcourt toutes les images
            for i in range(H): # indices de l'entrée participant au résultat
                for j in range(W):
                    for k in range(HH): # indices du filtre
                        for l in range(WW):
                          if i+k<H and j+l<W:
                                grad_input[n,i,j] += grad_output[n, i+k, j+l] * w[HH-k-1,WW-l-1]

    return grad_input, grad_w


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                     dense layer
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def init_dense_layer_weights(input_size, output_size):
  """
  INSPIRED BY:  https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9

  initialize the weights of a dense layer
  """
  weights = np.random.normal(loc=0.0,scale = np.sqrt(2/(input_size+output_size)),
                                        size = (input_size,output_size))
  return weights

def dense_forward(input,weights):
  """
  INSPIRED BY:  https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9

  forward phase of a dense layer
  """
  cache = (input,weights)
  output = np.dot(input,weights)

  return output, cache

def dense_backward(grad_output,cache):
  """
  INSPIRED BY:  https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9

  backward phase of a dense layer. The cache was returned by dense_forward.
  """
  input,weights = cache

  grad_input = np.dot(grad_output, weights.T)

  grad_weights = np.dot(input.T, grad_output)

  return grad_input , grad_weights


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                     ReLU layer
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ReLU_forward(input):
  """
  INSPIRED BY:  https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
  """

  cache = (input)
  output = np.maximum(0,input)
  return output , cache

def ReLU_backward(grad_output,cache):
  """
  INSPIRED BY:  https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
  """

  input = cache
  relu_grad = input > 0
  grad_input=grad_output*relu_grad

  return grad_input


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                     logistic function
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Logistic_function(input):
  return 1 / (1 + np.exp(-input))

def derivative_Logistic_function(input):
  lf_input=Logistic_function(input)
  return lf_input*(1-lf_input)



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                     MAIN
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#1) INIT

#get and treat data
data=np.genfromtxt('./mnist_0_1.csv',delimiter=',')

n_tot=data.shape[0]
p=data.shape[1]

y_train=data[:int(2.*n_tot/3.),0].reshape(-1,1)
X_train=(data[:int(2.*n_tot/3.),1:]/(255.*p)).reshape(-1,28,28)

y_test=data[int(2.*n_tot/3.):,0].reshape(-1,1)
X_test=(data[int(2.*n_tot/3.):,1:]/(255.*p)).reshape(-1,28,28)


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)




def ShowMNISTObservation(X_data,y_data,obsNb=0):
  plt.clf()
  plt.imshow(X_data[obsNb,:].reshape((28,28)))
  plt.title('Observation '+str(obsNb)+': Label '+str((y_data[obsNb,0])))
  plt.show()



#ShowMNISTObservation(X_train,y_train,0)
#ShowMNISTObservation(X_train,y_train,1)
#...
#ShowMNISTObservation(X_test,y_train,1)





#2) TRAINING
#generate the network parameters

w_conv=init_convolution_filter([5,5])
w_dense=init_dense_layer_weights(28*28, 1)

w_conv_init=w_conv.copy()
w_dense_init=w_dense.copy()

#Stochastic gradient Descent
Batch_size=100
nb_epochs=2

n=X_train.shape[0]

total_time_in_conv_backward=0.
start_time= time()

for epoch in range(nb_epochs):
    print('epoch:',epoch)
    obsIDs=np.arange(n)
    np.random.shuffle(obsIDs)
    batch_start=0

    while batch_start+Batch_size<n:
      x_batch=X_train[obsIDs[batch_start:batch_start+Batch_size],:,:]
      y_true_batch=y_train[obsIDs[batch_start:batch_start+Batch_size],:]

      #forward phase
      output_1 , cache_1 = conv_forward(x_batch, w_conv)
      output_2 , cache_2 = ReLU_forward(output_1)
      output_3 , cache_3 = dense_forward(output_2.reshape(-1,28*28),w_dense) #the reshape is used to flatten the image
      y_pred = Logistic_function(output_3)

      MSE_loss = np.mean( np.power(y_pred-y_true_batch,2.) ) #log-likelihood would be slightly more general

      #backward phase
      grad_MSE_loss = 2*(y_pred-y_true_batch)
      grad_output_3=grad_MSE_loss*derivative_Logistic_function(output_3)
      grad_output_2 , grad_w_dense = dense_backward(grad_output_3,cache_3)
      grad_output_1=ReLU_backward(grad_output_2.reshape(-1,28,28),cache_2)  #the reshape reverts the flattening
      rtime = time()
      grad_input, grad_w_conv=conv_backward(grad_output_1, cache_1)
      rtime = time() - rtime
      total_time_in_conv_backward+=rtime

      #gradient descent-update
      w_dense-=500*grad_w_dense
      w_conv-=0.001*grad_w_conv    #REMARK: the learning rate has to be much smaller on the convolutional layer than on the dense layer (the data should be re-scaled in this layer to avoid this)

      #prepare the next mini-batch
      batch_start+=Batch_size

      print(MSE_loss)

print('Total time:',time() - start_time)
print('Time in conv_backward:',total_time_in_conv_backward)


#3) TEST

#show the trained information

plt.imshow(w_dense_init.reshape((28,28)))
plt.title('initial dense layer')
plt.show()

plt.imshow(w_dense.reshape((28,28)))
plt.title('Trained dense layer')
plt.show()

print('Initial convolution filter:\n',w_conv_init)
print('Trained convolution filter:\n',w_conv)


#predictions on test data

output_1 , cache_1 = conv_forward(X_test, w_conv)
output_2 , cache_2 = ReLU_forward(output_1)
output_3 , cache_3 = dense_forward(output_2.reshape(-1,28*28),w_dense)
y_pred = Logistic_function(output_3)

MSE_loss = np.mean( np.power(y_pred-y_test,2.) )

Nb_false_pred=np.sum(np.abs(1*(y_pred>0.5)-y_test))

prct_false_pred=100.*Nb_false_pred/y_test.shape[0]

print('Percentage of good predictions:',100-prct_false_pred)

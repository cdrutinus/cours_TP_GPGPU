"""
If using google colab:
* Click on Runtime (excecution) and select Change runtime type (modifier le type d'excecution).
  Then select GPU in Hardware Acceleration (accélérateur matériel)
* Start your session by installing pycuda with the command:
  -> !pip install pyopencl
"""

import pyopencl as cl
import sys


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#function to get device information
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def output_device_info(device_id):
    print("Device is "+device_id.name)

    if device_id.type == cl.device_type.GPU:
        print("GPU from "+ device_id.vendor)
    elif device_id.type == cl.device_type.CPU:
        print("CPU from "+ device_id.vendor)
    else:
        print("non CPU of GPU processor from "+ device_id.vendor)
    print("Max of "+str(device_id.max_compute_units)+" compute units")





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#EXERCICE : LE BUT ICI EST D'EXPLORER LES CONFIGURATIONS
#POSSIBLES D'UN CONTEXTE DE CALCUL SUR VOTRE MACHINE
#-> N'hésitez pas à changer les choix de platforms et devices
#pour trouver la meilleure configuration possible pour des
#calculs parallèles.
#-> Vous pourrez vous aider de la fonction output_device_info
#que vous pourrez éventuellement étendre avec d'autres infos
#t.q. "max_work_item_dimensions", "max_work_group_size", ...
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



platforms = cl.get_platforms()
print(platforms)


devices = platforms[0].get_devices()
print(devices)

output_device_info(devices[0])










#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#REMARQUE POUR LA SUITE :
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#On créer ensuite un contexte
#  -> soit en spécifiant le device avec :

context = cl.Context([devices[1]])


#  -> soit de manière purement automatique avec :

context = cl.create_some_context()


# -> soit de manière automatique après avoir défini la variable d'environnement PYOPENCL_CTX (exemple PYOPENCL_CTX='1:0') avec :

context = cl.create_some_context()

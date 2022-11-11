import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def init_plt():
    
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 18
    
def init_tf():
    tf.keras.backend.set_floatx('float64')
    pass
    #physical_devices = tf.config.list_physical_devices('GPU') 
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.keras.backend.set_floatx('float64')
    
def dim(x, i): 
    return np.expand_dims(x, i)

def tensor(x):
    return tf.convert_to_tensor(x)

def list_to_tensors(l):
    return list(map(tensor, l))

def dict_to_tensors(d):
    return {k: tensor(v) for k,v in d.items()}
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import models, layers

def plot_model(model, to_file = 'model.png', expand_nested = True):
    return tf.keras.utils.plot_model(model, show_shapes = True, show_dtype = True, expand_nested = expand_nested, to_file = to_file)

class Generator:

    def __init__(self):

        self.KERNEL_SIZE = 31
        self.STRIDE = 2 
        self.layer = {
            'conv': 0,
            'ac': 0,
            'deconv': 0
        }

    def get_layer_num(self, l_type, inc = True):

        if inc:
            self.layer[l_type] += 1 
        return self.layer[l_type]


    def get_layer_name(self, l_type):

        l_num = self.get_layer_num(l_type = l_type)
        return 'gen_' + l_type + '_' + str(l_num)


    def gen_block(self, n_filters, prev_input, padding = 'same', block_type = 'conv', activation = 'prelu'):
        
        if block_type == 'conv':
            conv = layers.Conv1D(
                filters = n_filters, kernel_size = self.KERNEL_SIZE, strides = self.STRIDE, 
                padding = padding, name = self.get_layer_name(l_type = 'conv'), 
                kernel_regularizer = tf.keras.regularizers.l1(l1 = 100)
            )(prev_input)
        else:
            conv = layers.Conv1DTranspose(
                filters = n_filters, kernel_size = self.KERNEL_SIZE, strides = self.STRIDE, 
                padding = padding, name = self.get_layer_name(l_type = 'deconv'), 
                kernel_regularizer = tf.keras.regularizers.l1(l1 = 100)
            )(prev_input)
        
        if activation == 'prelu':
            ac = layers.PReLU(name = self.get_layer_name(l_type = 'ac'))(conv)
        
        elif activation == 'tanh':
            ac = layers.Activation('tanh', name = self.get_layer_name(l_type = 'ac'))(conv)
        
        else:
            raise ValueError("Invalid Activation. Possible activations: prelu, tanh")

        return ac 


    def construct(self, input_shape = [16384, 1]):

        inp = layers.Input(shape = input_shape)

        b1 = self.gen_block(n_filters = 16, prev_input = inp, block_type = 'conv')
        b2 = self.gen_block(n_filters = 32, prev_input = b1, block_type = 'conv')
        b3 = self.gen_block(n_filters = 32, prev_input = b2, block_type = 'conv')
        b4 = self.gen_block(n_filters = 64, prev_input = b3, block_type = 'conv')
        b5 = self.gen_block(n_filters = 64, prev_input = b4, block_type = 'conv')
        b6 = self.gen_block(n_filters = 128, prev_input = b5, block_type = 'conv')
        b7 = self.gen_block(n_filters = 128, prev_input = b6, block_type = 'conv')
        b8 = self.gen_block(n_filters = 256, prev_input = b7, block_type = 'conv')
        b9 = self.gen_block(n_filters = 256, prev_input = b8, block_type = 'conv')
        b10 = self.gen_block(n_filters = 512, prev_input = b9, block_type = 'conv')
        b11 = self.gen_block(n_filters = 1024, prev_input = b10, block_type = 'conv')

        concat = layers.Lambda(lambda x : tf.concat([x, tf.zeros_like(x)], axis = 2))(b11)

        b12 = self.gen_block(n_filters = 512, prev_input = concat, block_type = 'deconv')
        s1 = layers.add([b10, b12])
        
        b13 = self.gen_block(n_filters = 256, prev_input = s1, block_type = 'deconv')
        s2 = layers.add([b9, b13])
        
        b14 = self.gen_block(n_filters = 256, prev_input = s2, block_type = 'deconv')
        s3 = layers.add([b8, b14])
        
        b15 = self.gen_block(n_filters = 128, prev_input = s3, block_type = 'deconv')
        s4 = layers.add([b7, b15])
        
        b16 = self.gen_block(n_filters = 128, prev_input = s4, block_type = 'deconv')
        s5 = layers.add([b6, b16])

        b17 = self.gen_block(n_filters = 64, prev_input = s5, block_type = 'deconv')
        s6 = layers.add([b5, b17])

        b18 = self.gen_block(n_filters = 64, prev_input = s6, block_type = 'deconv')
        s7 = layers.add([b4, b18])

        b19 = self.gen_block(n_filters = 32, prev_input = s7, block_type = 'deconv')
        s8 = layers.add([b3, b19])

        b20 = self.gen_block(n_filters = 32, prev_input = s8, block_type = 'deconv')
        s9 = layers.add([b2, b20])

        b21 = self.gen_block(n_filters = 16, prev_input = s9, block_type = 'deconv')
        s10 = layers.add([b1, b21])

        b22 = self.gen_block(n_filters = 1, prev_input = s10, block_type = 'deconv', activation = 'tanh')

        gan = models.Model(inputs = inp, outputs = b22)
        return gan




class Discriminator:

    def __init__(self):
        self.KERNEL_SIZE = 31
        self.STRIDE = 2 
        self.layer = {
            'conv': 0,
            'bnorm': 0, 
            'ac': 0,
        }


    def get_layer_num(self, l_type, inc = True):

        if inc:
            self.layer[l_type] += 1 
        return self.layer[l_type]


    def get_layer_name(self, l_type):

        l_num = self.get_layer_num(l_type = l_type)
        return 'disc_' + l_type + '_' + str(l_num)


    def disc_block(self, n_filters, prev_input, padding = 'same'):
        
        conv = layers.Conv1D(
            filters = n_filters, kernel_size = self.KERNEL_SIZE, strides = self.STRIDE, 
            padding = padding, name = self.get_layer_name(l_type = 'conv'), 
            kernel_regularizer = tf.keras.regularizers.l1(l1 = 100)
        )(prev_input)

        bnorm = layers.BatchNormalization(virtual_batch_size = 5, name = self.get_layer_name(l_type = 'bnorm'))(conv)        
        
        ac = layers.LeakyReLU(alpha = 0.3, name = self.get_layer_name(l_type = 'ac'))(bnorm)
        return ac


    def construct(self, input_shape = [16384, 1]):

        inp = layers.Input(shape = input_shape)

        b1 = self.disc_block(n_filters = 16, prev_input = inp)
        b2 = self.disc_block(n_filters = 32, prev_input = b1)
        b3 = self.disc_block(n_filters = 32, prev_input = b2)
        b4 = self.disc_block(n_filters = 64, prev_input = b3)
        b5 = self.disc_block(n_filters = 64, prev_input = b4)
        b6 = self.disc_block(n_filters = 128, prev_input = b5)
        b7 = self.disc_block(n_filters = 128, prev_input = b6)
        b8 = self.disc_block(n_filters = 256, prev_input = b7)
        b9 = self.disc_block(n_filters = 256, prev_input = b8)
        b10 = self.disc_block(n_filters = 512, prev_input = b9)
        b11 = self.disc_block(n_filters = 1024, prev_input = b10)
    
        compress = layers.Conv1D(filters = 1, kernel_size = 1, padding = 'same', name = self.get_layer_name(l_type = 'conv'))(b11)

        flatten = layers.Flatten(name = 'flatten')(compress)
        binary = layers.Dense(1, activation = 'sigmoid', name = 'binary_classifier')(flatten)

        disc = models.Model(inputs = inp, outputs = binary)
        return disc

import numpy as np 
from matplotlib import pyplot as plt 

import librosa 

import tensorflow as tf 
from tensorflow.keras import models, layers 
from tensorflow.keras import backend as K 

import os 
from tqdm import tqdm 

from segan_utils import Generator, Discriminator 

speech_path = 'Dataset/Speech'
egg_path = 'Dataset/EGG'

def load_reals():
    
    names = [i for i in tqdm(os.listdir(speech_path)) if i in os.listdir(egg_path)]
    return names


def generate_outputs(names, batch_size):

    idx = np.random.randint(0, len(names), batch_size).tolist()
    outputs = [librosa.load(os.path.join(egg_path, names[i]))[0] for i in idx]
    return outputs


def generate_inputs(names, batch_size):
    #load inputs for generator
    idx = np.random.randint(0, len(names, batch_size).tolist()
    inputs = [librosa.load(os.path.join(speech_path, names[i]))[0] for i in idx]
    return inputs


def generate_fakes(gen, names, batch_size):
    #generate fake samples with generator
    gen_input = generate_inputs(names, batch_size)
    return gen(gen_input)


class GAN:

    def __init__(self):

        self.input_shape = [16384, 1]
        self.batch_size = 400
        self.lr = 0.0002
        self.gan = None
        self.gen = None 
        self.disc = None
        self.epochs = 100 

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def get_gan(self):
        return self.gan
    
    def create_gan(self):

        self.gen = Generator().construct()
        
        self.disc = Discriminator().construct()
        self.disc.compile(loss = self.wasserstein_loss, optimizer = 'adam')
        self.disc.trainable = False 

        self.gan = models.Sequential([self.gen, self.disc]) 

        self.gan.compile(loss = self.wasserstein_loss, optimizer = 'adam')
 
    def train_gan(self):

        gen, disc = selg.gan.layers 

        names = load_reals()
        n_batches = len(names)//self.batch_size

        for epoch in range(self.epochs):

            for i in trange(n_batches, desc = 'Epoch {}/{}'.format(epoch + 1, self.epochs), ncols = 100):

                disc.trainable = False 

                X_real = generate_outputs(names, self.batch_size)
                y_real = np.ones((self.batch_size, 1))

                X_fake = generate_fakes(gen, names, self.batch_size)
                y_fake = np.zeros((self.batch_size, 1)) 

                disc_loss_real, _ = disc.train_on_batch(X_real, y_real)
                disc_loss_fake, _ = disc.train_on_batch(X_fake, y_fake) 

                disc.trainable = True 

                ... 

        



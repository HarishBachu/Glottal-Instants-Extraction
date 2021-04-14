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

def generate_reals(names, batch_size):

    idx = np.random.randint(0, len(names), batch_size).tolist()
    inputs = 
    #return real samples

def generate_latents():
    ...
    #load inputs for generator

def generate_fakes():
    ...
    #generate fake samples with generator


class GAN:

    def __init__(self):

        self.input_shape = [16384, 1]
        self.batch_size = 400
        self.lr = 0.0002
        self.gan = None
        self.gen = None 
        self.disc = None

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def create_gan(self):

        self.gen = Generator().construct()
        self.disc = Discriminator().construct()
        self.disc.compile(loss = self.wasserstein_loss, optimizer = 'adam')

        self.gan = models.Sequential()
        self.gan.add(self.gen)
        self.gan.add(self.disc) 

        self.gan.compile(loss = self.wasserstein_loss, optimizer = 'adam')
 
        return self.gan 

import numpy as np 
from matplotlib import pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import models, layers 
from tensorflow.keras import backend as K 

import os 
from tqdm import tqdm 

from segan_utils import Generator, Discriminator 

def load_reals():
    ...
    #return real outputs

def generate_reals():
    ...
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

        self.gen = Generator() 
        self.disc = Discriminator() 
        self.disc.compile(loss = self.wasserstein_loss, optimizer = ...)

        self.gan = models.Sequential()
        self.gan.add(self.gen)
        self.gan.add(self.disc) 

        self.gan.compile(loss = self.wasserstein_loss, optimizer = ...)



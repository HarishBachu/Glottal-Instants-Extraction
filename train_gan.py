import numpy as np 
from matplotlib import pyplot as plt 

import librosa 

import tensorflow as tf 
from tensorflow.keras import models, layers 
from tensorflow.keras import backend as K 

import os 
from tqdm import tqdm, trange

from segan_utils import Generator, Discriminator 

speech_path = 'Dataset/Speech'
egg_path = 'Dataset/EGG'

def load_reals():
    
    names = [i for i in tqdm(os.listdir(speech_path)) if i in os.listdir(egg_path)]
    return names


def generate_outputs(names, batch_size, trim_size):

    idx = np.random.randint(0, len(names), batch_size).tolist()
    outputs = []
    for i in idx:
        x = librosa.load(os.path.join(egg_path, names[i]))[0]
        a = len(x)
        lim = a - trim_size 
        low = np.random.randint(0, lim)
        high = low + trim_size 
        x = x[low:high]
        outputs.append(np.expand_dims(x, -1))

    # outputs = [librosa.load(os.path.join(egg_path, names[i]))[0] for i in idx]
    return np.asarray(outputs)


def generate_inputs(names, batch_size, trim_size):
    #load inputs for generator
    idx = np.random.randint(0, len(names), batch_size).tolist()
    inputs = []
    for i in idx:
        x = librosa.load(os.path.join(speech_path, names[i]))[0]
        a = len(x)
        lim = a - trim_size 
        low = np.random.randint(0, lim)
        high = low + trim_size 
        x = x[low:high]
        inputs.append(np.expand_dims(x, -1))
    # inputs = [librosa.load(os.path.join(speech_path, names[i]))[0] for i in idx]
    return np.asarray(inputs)


def generate_fakes(gen, names, batch_size, trim_size):
    #generate fake samples with generator
    gen_input = generate_inputs(names, batch_size, trim_size)
    return gen.predict(gen_input)


class GAN:

    def __init__(self):

        self.input_shape = [16384, 1]
        self.batch_size = 40
        self.trim_size = 16384
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

        self.gen = Generator().construct(batch_size = self.batch_size)
        
        self.disc = Discriminator().construct()
        self.disc.compile(loss = 'mse', optimizer = 'adam')
        self.disc.trainable = False 

        self.gan = models.Sequential([self.gen, self.disc]) 

        self.gan.compile(loss = 'mse', optimizer = 'adam')
 
    def train_gan(self):

        gen, disc = self.gan.layers 

        names = load_reals()
        n_batches = len(names)//self.batch_size

        for epoch in range(self.epochs):

            for i in trange(n_batches, desc = 'Epoch {}/{}'.format(epoch + 1, self.epochs), ncols = 100):

                disc.trainable = True

                X_real = generate_outputs(names, self.batch_size, self.trim_size)
                y_real = np.ones((self.batch_size, 1))

                X_fake = generate_fakes(gen, names, self.batch_size, self.trim_size)
                y_fake = np.zeros((self.batch_size, 1)) 

                disc_loss_real, _ = disc.train_on_batch(X_real, y_real)
                disc_loss_fake, _ = disc.train_on_batch(X_fake, y_fake) 

                disc.trainable = False 

                X_gan = generate_inputs(names, self.batch_size, self.trim_size)
                y_gan = np.ones((self.batch_size, 1))

                gan_loss, gan_acc = self.gan.train_on_batch(X_gan, y_gan)

            
        



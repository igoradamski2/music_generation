#!../testenv/bin/python3

import pretty_midi
import midi
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Concatenate, LSTM

from keras.optimizers import Adam

from keras import backend as K

import tensorflow as tf
#import tensorflow_probability as tfp # for tf version 2.0.0, tfp version 0.8 is needed 
import numpy as np

import csv
from sys import stdout
import random

# My code
from loading import *
from models import *
from data import *
from midi_to_statematrix import *


def my_binary_loss_seq(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, 78])
    y_pred = tf.reshape(y_pred, [-1, 78])
    
    bce = tf.keras.losses.BinaryCrossentropy()
    
    return bce(y_true, y_pred)

def generate(train_batch):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        new_batch    = next(train_batch)
        new_batch.target_split = 50
        new_batch.featurize(use_biaxial = False, out_seq = True)
        yield ([tf.convert_to_tensor(new_batch.context, dtype = tf.float32), 
                tf.convert_to_tensor(new_batch.target_train, dtype = tf.float32)], 
               tf.convert_to_tensor(new_batch.target_pred, dtype = tf.float32))

if __name__ == '__main__':

	print("TensorFlow version: {}".format(tf.__version__))
	print("GPU is available: {}".format(tf.test.is_gpu_available()))

	file = 'maestro-v2.0.0/maestro-v2.0.0.csv'

	# Call data class
	data = DataObject(file, what_type = 'train', train_tms = 100, test_tms = 100, fs = 20, window_size = 15)


	# Create a batch class which we will iterate over
	train_batch = Batch(data, batch_size = 32, songs_per_batch = 4)

	curr_batch = train_batch.data
	curr_batch.target_split = 50
	curr_batch.featurize(use_biaxial = False, out_seq = True)

	model = biaxial_target_model(curr_batch, encoder_output_size = 20)
	model.compile(loss = my_binary_loss_seq, optimizer = Adam(learning_rate=0.0004))

	model.summary()

	history = model.fit_generator(
                    generate(train_batch),
                    steps_per_epoch=1024,
                    epochs=10)



	# dd/mm/YY
	filename = 'biaxial_nofeat_seq_pred_5epoch'

	model.save_weights(filename+'.h5')

	with open(filename+'.txt', 'w+') as f:
		for element in history.history['loss']:
			f.write('\n'+str(element))
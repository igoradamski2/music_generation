import os
import pretty_midi
import midi
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Concatenate, LSTM
from keras.optimizers import Adam

from keras import backend as K

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import csv
from sys import stdout
import random
import scipy.stats as st
from os import path
import sys

import argparse

# My code
from loading import *
from models import *
from data import *
from midi_to_statematrix import *

##################### GENERATION PARAMETERS #####################
##################### READ THEM FROM A JSON #####################

my_model_name = "biaxial_pn_encoder_concat_deeplstm_cont.h5"
foldername = 'experiment_switch_order3'

# data 
file = 'maestro-v2.0.0/maestro-v2.0.0.csv'
what_type = 'test'
train_tms = 40
test_tms = 20
batch_size = 64
songs_per_batch = 16
seed = 1212

# turn probabilities to notes params
how = 'random'
normalize = False
remap_to_max = True
turn_on_notes = 8
divide_prob = 2
articulation_prob = 0.0018
remap_prob = 0.35

# Recurrence params
pick_pred_from_idx = 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Songs for Website')

	parser.add_argument('-GPU', dest = "use_gpu", default = False, 
                        help='Use GPU')
	parser.add_argument('-bs','--bath', help='batch size')

	args = vars(parser.parse_args())

	print("TensorFlow version: {}".format(tf.__version__))
	print("GPU is available: {}".format(tf.test.is_gpu_available()))

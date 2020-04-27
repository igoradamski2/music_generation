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



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Songs for Website')

	parser.add_argument('-GPU', dest = "use_gpu", default = False, 
                        help='Use GPU')
	parser.add_argument('-bs','--bath', help='batch size')

	args = vars(parser.parse_args())

	print("TensorFlow version: {}".format(tf.__version__))
	print("GPU is available: {}".format(tf.test.is_gpu_available()))

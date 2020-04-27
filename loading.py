import pretty_midi
import numpy as np
#import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Concatenate

from keras import backend as K

import tensorflow as tf
#import tensorflow_probability as tfp # for tf version 2.0.0, tfp version 0.8 is needed 
import numpy as np

import csv
from sys import stdout
import random
import copy

from midi_to_statematrix import *
from data import *


class DataLinks(object):
    
    def __init__(self, file, what_type, train_tms, test_tms):
        self.file = file
        self.what_type = what_type
        self.train_tms = train_tms
        self.test_tms = test_tms
        self.get_links()
        self.get_number_of_examples()
    
    def get_links(self):
        links    = []
        duration = []
        names    = []
        with open(self.file, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if row[2] == self.what_type:
                        links.append(row[4])
                        duration.append(float(row[-1]))
                        names.append(row[0])
                    line_count += 1
        self.links = links
        self.duration = duration
        self.names = names
    
    def get_number_of_examples(self):
        num_examples   = []
        example_length = self.train_tms + self.test_tms 
        for idx, link in enumerate(self.links):
            link_duration = self.duration[idx]
            num_examples.append(int(float(link_duration)/example_length))
        
        self.num_examples = num_examples
    

class TrainingExample(object):
    
    def __init__(self, context, target, link, start, target_split, window_size, test_tms):
        self.context = context
        self.target  = target
        self.target_split = target_split
        self.link    = link
        self.start   = start
        self.window_size = window_size
        self.test_tms = test_tms

    def __len__(self):
        return len(self.link)

    def featurize(self, use_biaxial = True, out_seq = True):

        if use_biaxial == True:
            
            # Featurize context

            self.context = np.transpose(self.context, axes = [1,0,2,3])
            

            # Featurize target

            # target shape is (batch_size, test_tms, 78, 2)

            # Firsly split on target split
            self.target_train = self.target[:,self.target_split:(self.target_split+self.window_size),:]
            self.target_pred  = self.target[:,(self.target_split+1):(self.target_split+self.window_size+1),:]

            tt_shape = self.target_train.shape

            # Now extract features with hexandrias shitty code
            features = np.reshape(self.target_train, 
                [tt_shape[0]*tt_shape[1], 
                 tt_shape[2], 
                 tt_shape[3]]
            )

            features = np.array(noteStateMatrixToInputForm(features))
            features = np.reshape(features,
                [tt_shape[0],
                 tt_shape[1],
                 tt_shape[2],
                 features.shape[-1]])

            # Now get rid of articulation
            self.target_train = DataObject.drop_articulation(self.target_train)
            self.target_pred  = DataObject.drop_articulation(self.target_pred)

            # Now add last change variable
            last_change = DataObject.get_last_change_tensor(DataObject.drop_articulation(self.target))
            last_change = last_change[:,self.target_split:(self.target_split+self.window_size),:]

            self.target_train = np.append(features, np.expand_dims(self.target_train, axis = 3), axis = 3)
            self.target_train = np.append(self.target_train, np.expand_dims(last_change, axis = 3), axis = 3)
            #self.target_train = np.insert(self.target_train, self.target_train.shape[3], self.target_split/100, axis=3)
            self.target_train = DataObject.add_time_information(self.target_train, start = self.target_split, size = self.test_tms)

        else:

            if out_seq == False:

                # Featurize context
                self.context = np.transpose(self.context, axes = [1,0,2,3])

                # Firsly split on target split
                self.target_train = self.target[:,0:self.target_split,:]
                self.target_pred  = self.target[:,self.target_split,:]

                # Now get rid of articulation
                self.target_train = DataObject.drop_articulation(self.target_train)
                self.target_pred = DataObject.drop_articulation3d(self.target_pred)

                # Now add last change variable
                last_change = DataObject.get_last_change_tensor(self.target_train)

                self.target_train = np.expand_dims(self.target_train, axis = 3)

                self.target_train = np.append(self.target_train, np.expand_dims(last_change, axis = 3), axis = 3)

                # Add time information
                self.target_train = DataObject.add_time_information(self.target_train, start = 0)

            elif out_seq == True:

                # Featurize context
                self.context = np.transpose(self.context, axes = [1,0,2,3])

                # Firsly split on target split
                self.target_train = self.target[:,0:-1,:]
                self.target_pred  = self.target[:,1:,:]

                # Now get rid of articulation
                self.target_train = DataObject.drop_articulation(self.target_train)
                self.target_pred = DataObject.drop_articulation(self.target_pred)

                # Now add last change variable
                last_change = DataObject.get_last_change_tensor(self.target_train)

                self.target_train = np.expand_dims(self.target_train, axis = 3)

                self.target_train = np.append(self.target_train, np.expand_dims(last_change, axis = 3), axis = 3)

                # Add time information
                self.target_train = DataObject.add_time_information(self.target_train, start = 0)




    def contextify(self, window_size):

        print("Here")

        """
        window_size - size of the window given in TIMESTEPS

        """

        timesteps  = self.context.shape[-1]
        stepsize   = int(window_size/3)
        batch_size = self.context.shape[0]

        assert timesteps > window_size, "window_size bigger than number of timesteps in context"

        idx = 0
        curr_window = self.context[:, :, idx:window_size]
        contextified = [curr_window]
        
        while True:
            idx += 1
            curr_window = self.context[:, :, (idx*stepsize):((idx*stepsize)+window_size)]
            if curr_window.shape[-1] < window_size:
                pad_number = window_size - curr_window.shape[-1]
                curr_window = tf.concat([curr_window, tf.zeros([batch_size,88,pad_number], dtype=tf.float32)], 2)
                contextified.append(curr_window)
                break
            else:
                contextified.append(curr_window)

        contextified = tf.convert_to_tensor(contextified, dtype=tf.float32) # [window_num, batch_size, note_size, timesteps]

        contextified = tf.transpose(contextified, perm = [1,0,3,2])

        #desired_shpe = [contextified.shape[1], # batch_size
        #                contextified.shape[0], # window_number 
        #                contextified.shape[-1], # timestep
        #                contextified.shape[2]] # [batch_size, window_number, timestep, note_size]
        #contextified = tf.reshape(contextified, desired_shpe)
        self.context = contextified

        #target_shape = [self.target.shape[0],
        #                self.target.shape[2],
        #                self.target.shape[1]]

        self.target = tf.transpose(self.target, perm = [0,2,1])



class Batch(object):

    def __init__(self, data_object, batch_size, songs_per_batch):

        assert isinstance(data_object, DataObject), "Pass an instance of DataObject to Batch"
        assert batch_size < len(data_object), "Batch size must be smaller than data length"
        assert batch_size % songs_per_batch == 0, "Select batch_size divisible by songs_per_batch"
        
        self.all_data        = data_object
        self.batch_size      = batch_size
        self.songs_per_batch = songs_per_batch 
        self.data            = data_object.generate_batch(batch_size, songs_per_batch)

        assert len(self.data) == self.batch_size, "Length of batch object is not batch_size"
        
    def __next__(self):
        
        self.data = self.all_data.generate_batch(self.batch_size, self.songs_per_batch)
        
        return self.data

    def __iter__(self):
        return self
        

class DataObject(DataLinks):

    def __init__(self, file, 
                 what_type, 
                 train_tms, 
                 test_tms,
                 fs,
                 window_size,
                 seed = None,
    ):
        super(DataObject, self).__init__(file, what_type, train_tms, test_tms)
        self.fs = fs
        self.window_size = window_size
        self.seed = seed

    #def __getitem__(self, arg):
    #    return DataObject(self.xdata[arg], self.ydata[arg])

    def __len__(self):
        return sum(self.num_examples)

    def generate_batch(self, batch_size, songs_per_batch):
        
        batch_data = []
        l_batch_data_context   = []
        r_batch_data_context   = []
        batch_data_target      = []
        batch_data_link        = []
        batch_data_starts      = []
        
        random.seed(self.seed)
        random_songs       = random.sample(self.links, songs_per_batch)
        random_songs_index = [self.links.index(song) for song in random_songs]
        
        examples_per_song = batch_size/songs_per_batch

        random.seed(self.seed)
        target_split = random.randint(0, (self.test_tms-self.window_size - 1))
        
        for idx, link in enumerate(random_songs):
            
            piano_matrix = DataObject.get_piano_matrix(self, link) # whole matrix
            timesteps = piano_matrix.shape[0]
            #fs        = int(timesteps/self.duration[random_songs_index[idx]])
            
            for i in range(int(examples_per_song)):
                
                random.seed(self.seed + i)
                start        = random.randint(self.train_tms, timesteps-(self.train_tms+self.test_tms))
                
                l_batch_data_context.append(piano_matrix[(start-self.train_tms):start, :])
                r_batch_data_context.append(piano_matrix[(start+self.test_tms):(start+self.train_tms+self.test_tms),:])
                batch_data_target.append(piano_matrix[start:(start+self.test_tms), :])
                batch_data_link.append(link)
                batch_data_starts.append(start)

        batch_data = TrainingExample(np.stack([DataObject.drop_articulation(np.array(l_batch_data_context)), 
                                               DataObject.drop_articulation(np.array(r_batch_data_context))]),
                                     np.array(batch_data_target),
                                     batch_data_link,
                                     batch_data_starts,
                                     target_split,
                                     self.window_size,
                                     self.test_tms,
        )
        
        return batch_data
                    
    def get_piano_matrix(self, link, stripSilence = True):
    
        piano_matrix = np.array(midiToNoteStateMatrix('maestro-v2.0.0/'+link))

        #old_piano_matrix        = np.array(piano_matrix)
        #old_piano_matrix[:,:,0] = old_piano_matrix[:,:,0]+old_piano_matrix[:,:,1]
        #piano_matrix = np.zeros((old_piano_matrix.shape[0], old_piano_matrix.shape[1]))
        #piano_matrix[:,:] = old_piano_matrix[:,:,0]
        #piano_matrix[piano_matrix > 0] = 1

        if stripSilence == True:
            # Strip silence at beginning and end
            start = np.min(np.where(piano_matrix != 0)[0])
            end   = np.max(np.where(piano_matrix != 0)[0])

            piano_matrix = piano_matrix[start:end, :]
            
        return piano_matrix

    @staticmethod
    def drop_articulation(tensor):

        """
        this method is only for 4-dimensional tensors
        """

        old_piano_matrix        = copy.deepcopy(tensor)
        old_piano_matrix[:,:,:,0] = old_piano_matrix[:,:,:,0]+old_piano_matrix[:,:,:,1]
        piano_matrix = np.zeros((old_piano_matrix.shape[0], 
                                 old_piano_matrix.shape[1],
                                 old_piano_matrix.shape[2]))
        piano_matrix[:,:,:] = old_piano_matrix[:,:,:,0]
        piano_matrix[piano_matrix > 0] = 1

        return piano_matrix

    @staticmethod
    def drop_articulation3d(tensor):

        """
        this method is only for 3-dimensional tensors
        """

        old_piano_matrix        = copy.deepcopy(tensor)
        old_piano_matrix[:,:,0] = old_piano_matrix[:,:,0]+old_piano_matrix[:,:,1]
        piano_matrix = np.zeros((old_piano_matrix.shape[0], 
                                 old_piano_matrix.shape[1]))
        piano_matrix[:,:] = old_piano_matrix[:,:,0]
        piano_matrix[piano_matrix > 0] = 1

        return piano_matrix

    @staticmethod
    def get_last_change_tensor(tensor):

        """
        tensor must be shape [batch_size, time, note]
        """

        look_back   = np.roll(copy.deepcopy(tensor), shift=1, axis=1)
        is_the_same = look_back == tensor

        change = np.zeros(is_the_same.shape)

        for time in range(1,tensor.shape[1]):
            curr_change = is_the_same[:,time,:]
            x_p,y_p = np.where(curr_change == True)
            x_0,y_0 = np.where(curr_change == False)
            change[x_p,time,y_p] = change[x_p,time-1,y_p]+1
            change[x_0,time,y_0] = 0

        return change

    @staticmethod
    def add_time_information(tensor, start = 0, size = 100):

        timestep = (np.arange(1,(tensor.shape[1])+1) + start)/size
        timestep = np.tile(timestep, (tensor.shape[0], tensor.shape[2], 1)) 
        timestep = np.transpose(timestep, [0,2,1])

        tensor = np.append(tensor, np.expand_dims(timestep, axis = 3), axis = 3) 

        return tensor      







        
    

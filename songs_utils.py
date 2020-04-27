import numpy as np
from sys import stdout
import copy
from tqdm import tqdm
import json
import os
#from midi2audio import FluidSynth

from loading import *
from models import *
import midi_to_statematrix

class FakeSong(DataLinks):
    def __init__(self, 
                 file, 
                 what_type,
                 link, 
                 name,
                 contextLength, 
                 start, 
                 length,
                 numRegions,
                 regionStarts,
                 regionLengths):

        super(FakeSong, self).__init__(file, what_type, 1, 1)
        self.link  = link
        self.name  = name
        self.contextLength = contextLength
        self.start = start
        self.length = length
        self.numRegions = numRegions
        self.regionStarts = regionStarts
        self.regionLengths = regionLengths

        # Figure out which link we are looking at
        link_index = self.links.index(self.link)
        self.duration = self.duration[link_index]
        del self.links

        # Get training examples
        self.getTrainingExamples()
    
    def getTrainingExamples(self):

        piano_matrix = DataObject.get_piano_matrix(self, self.link, stripSilence=False) # whole song
        timesteps = piano_matrix.shape[0]

        self.get_full_song_pianoroll(piano_roll=piano_matrix)

        self.getTargetIndices()

        self.trainingExamples = []

        for i in range(self.numRegions):

            # Get starts and ends in terms of timesteps
            start = int(np.floor(timesteps * (self.regionStarts[i]/self.duration)))
            end   = int(start + np.floor(timesteps * (self.regionLengths[i]/self.duration)) + 1)

            # Get the matrices
            l_context = np.expand_dims(piano_matrix[0:start, :], 0)
            r_context = np.expand_dims(piano_matrix[end:, :], 0)
            target    = np.expand_dims(piano_matrix[start:end, :], 0)

            assert (target.shape[1] == (self.targetIdx[i][1] - self.targetIdx[i][0])), "Indices are too small"

            # Cut contexts
            min_context_length = min(l_context.shape[1],
                                     r_context.shape[1],
                                     self.contextLength)

            l_context = l_context[:, -min_context_length:, :]
            r_context = r_context[:, 0:min_context_length, :]

            # Populate the training example
            self.trainingExamples.append(TrainingExample(
                np.stack([DataObject.drop_articulation(np.array(l_context)), 
                          DataObject.drop_articulation(np.array(r_context))]
                        ),
                np.array(target),
                0,                      # target_split
                self.link,              # link
                0,                      # start
                self.regionLengths[i],  # window_size
                self.regionLengths[i],  # test_tms,
                )
            )
    
    def get_full_song_pianoroll(self, piano_roll = None):

        if piano_roll is None:
            piano_roll = DataObject.get_piano_matrix(self, self.link, stripSilence=False) # whole song
        
        timesteps = piano_roll.shape[0]

        self.sec_per_timestep = self.duration/timesteps

        start = int(np.floor(timesteps * (self.start/self.duration)))
        end   = int(start + np.floor(timesteps * (self.length/self.duration)))

        self.full_song_pianoroll = DataObject.drop_articulation3d(piano_roll[start:end])

    def getTargetIndices(self):

        if not hasattr(self, 'full_song_pianoroll'):
            self.get_full_song_pianoroll()

        timesteps = self.full_song_pianoroll.shape[0]
        self.targetIdx = []

        for i in range(self.numRegions):
            start = int(np.floor((self.regionStarts[i] - self.start) * timesteps/self.length))
            end   = int(np.floor(start + self.regionLengths[i] * timesteps/self.length)) + 1

            self.targetIdx.append((start, end))
    
    def generate_patch(self, trainingExampleId, params):

        curr_test_batch = copy.deepcopy(self.trainingExamples[trainingExampleId])

        final_output = np.zeros((params.batch_size,       # batch size
                                params.lookBack+self.trainingExamples[trainingExampleId].target.shape[1]+params.lookBack, #timesteps
                                78))                      # note size

        # Populate from the front
        final_output[:,0:params.lookBack,:] = curr_test_batch.context[0,:,-params.lookBack:,:]

        # Populate from the back 
        final_output[:,-params.lookBack:,:] = curr_test_batch.context[1,:,0:params.lookBack,:]

        curr_test_batch.target[:,0:params.lookBack,:,0] = final_output[:,0:params.lookBack,:]
        curr_test_batch.target[:,0:params.lookBack,:,1] = np.zeros(final_output[:,0:params.lookBack,:].shape) # can be done as we already remove articulation in context

        curr_test_batch.target_split = 0                    # Start
        curr_test_batch.window_size  = params.lookBack      # Window
        curr_test_batch.featurize(use_biaxial = True)

        model = self.load_model(params.my_model_name, curr_test_batch, biaxial_pn_encoder_concat_deeplstm)

        def take_prediction(t, steps, lookBack):
            if t<=lookBack:
                return np.arange(lookBack-t, lookBack, 1)
            elif (t>lookBack and t<=(steps-lookBack+1)):
                return np.arange(0,lookBack, 1)
            elif t>(steps-lookBack+1):
                return np.arange(0, steps-t+1)
            
        def output_indices(t, steps, lookBack):

            if t<=lookBack:
                return np.arange(lookBack, lookBack+t, 1)
            elif (t>lookBack and t<=(steps-lookBack+1)):
                return np.arange(t, t+lookBack, 1)
            elif t>(steps-lookBack+1):
                return np.arange(t, steps+1, 1)

        steps = params.lookBack + self.trainingExamples[trainingExampleId].target.shape[1] - 1
        for timestep in range(1,steps):
            
            stdout.write('\rtimestep {}/{}'.format(timestep, steps))
            stdout.flush()
            
            prediction = model.predict([tf.convert_to_tensor(curr_test_batch.context, dtype = tf.float32), 
                                        tf.convert_to_tensor(curr_test_batch.target_train, dtype = tf.float32)],
                                    steps = 1)[:,take_prediction(timestep, steps, params.lookBack),:]
            
            #prediction = np.random.rand(*curr_test_batch.target_train.shape[:-1])[:,take_prediction(timestep, steps, params.lookBack),:]

            notes = np.zeros(prediction.shape)
            
            turn_on = [params.turn_on_notes]*params.batch_size
            indx = output_indices(timestep, steps, params.lookBack)
            for t in range(notes.shape[1]):

                articulation = np.multiply(prediction[:,t,:], final_output[:,indx[t]-1,:])
                articulation[articulation >= params.articulation_prob] = 1
                articulation[articulation < params.articulation_prob] = 0
                articulated_notes = np.sum(articulation, axis = -1)
                
                play_notes = self.turn_probabilities_to_notes(prediction[:,t,:], 
                                                turn_on = turn_on - articulated_notes,
                                                remap_prob = params.remap_prob,
                                                how = 'random', 
                                                normalize = params.normalize,
                                                divide_prob = params.divide_prob, 
                                                remap_to_max = params.remap_to_max)
                
                play_notes = play_notes + articulation
                play_notes[play_notes >= 1] = 1
                play_notes[play_notes < 1] = 0
                
                final_output[:,indx[t],:] = play_notes
            
            curr_test_batch = copy.deepcopy(self.trainingExamples[trainingExampleId])
            
            curr_test_batch.target[:,0:params.lookBack,:,0] = copy.deepcopy(final_output[:,timestep:(params.lookBack+timestep)])


            curr_test_batch.target_split = 0
            curr_test_batch.window_size  = params.lookBack
            curr_test_batch.featurize(use_biaxial = True)
        
        final_patch = final_output[:, params.lookBack:(params.lookBack+self.trainingExamples[trainingExampleId].target.shape[1]), :]

        return final_patch
    
    def fill_gaps(self, params):

        self.fake_song_pianoroll = copy.deepcopy(self.full_song_pianoroll)

        for patchId in tqdm(range(self.numRegions)):
            
            generated_patch = self.generate_patch(patchId, params)
            self.fake_song_pianoroll[self.targetIdx[patchId][0]:self.targetIdx[patchId][1]] = generated_patch[0] # 0th batch 

    def save_songs(self, name):

        assert os.path.isdir("samples/"+name) is False, "Pick different folder name (param: name), as this one exists"

        self.folderpath_ = "./samples/"+name
        os.mkdir(self.folderpath_)

        #fs = FluidSynth()
        self.full_song_pianoroll = np.append(np.expand_dims(self.full_song_pianoroll, axis = -1), 
                                             np.expand_dims(self.full_song_pianoroll, axis = -1), axis = -1)

        self.fake_song_pianoroll = np.append(np.expand_dims(self.fake_song_pianoroll, axis = -1), 
                                             np.expand_dims(self.fake_song_pianoroll, axis = -1), axis = -1)

        # Save the true song (in MIDI format)
        noteStateMatrixToMidi(self.full_song_pianoroll, name = (self.folderpath_ + "/true_song"))
        # and in FLAC format
        #fs.midi_to_audio((self.folderpath_ + "/true_song.mid"), (self.folderpath_ + "/true_song.flac"))

        # Save the fake song (in MIDI format)
        noteStateMatrixToMidi(self.fake_song_pianoroll, name = (self.folderpath_ + "/fake_song"))
        # and in FLAC format
        #fs.midi_to_audio((self.folderpath_ + "/fake_song.mid"), (self.folderpath_ + "/fake_song.flac"))

        # Save all the generation parameters
        save_data = {'link':self.link,
                     'start':self.start,
                     'end':self.start+self.length,
                     'numRegions':self.numRegions,
                     'regionStarts':self.regionStarts,
                     'regionLengths':self.regionLengths,
                     'realRegionStarts':[idx[0]*self.sec_per_timestep for idx in self.targetIdx],
                     'realRegionEnds':[idx[1]*self.sec_per_timestep for idx in self.targetIdx]}
        
        with open((self.folderpath_+'/song_metadata.json'), 'w+') as fp:
            json.dump(save_data, fp)


    @staticmethod
    def load_model(file, curr_batch, modelname, *modelparams):
        new_model = modelname(curr_batch, *modelparams)
    
        new_model.load_weights(file)
    
        return new_model

    @staticmethod
    def turn_probabilities_to_notes(prediction, 
                                    turn_on,
                                    remap_prob, 
                                    how = 'random', 
                                    normalize = True, 
                                    threshold = 0.1, 
                                    divide_prob = 2,
                                    remap_to_max = True):
        
        for batch in range(prediction.shape[0]):
            if turn_on[batch] <= 1:
                prediction[batch, :] = 0
                continue
            turn_off = prediction[batch, :].argsort()[:-int(turn_on[batch])]
            prediction[batch, :][turn_off] = 0
            
            if normalize: 
                prediction[batch, :] = st.norm.cdf((prediction[batch, :] - 
                                                    np.mean(prediction[batch, :][prediction[batch, :] > 0]))/
                                                np.sqrt(np.var(prediction[batch, :][prediction[batch, :]>0])))/divide_prob
                prediction[batch, :][turn_off] = 0
            
            if remap_to_max:
                prediction[batch, :] /= prediction[batch, :].max()
                prediction[batch, :] *= remap_prob
            
        if how == 'random':
            
            notes =  np.random.binomial(1, p=prediction)
            
        elif how == 'random_thresholded':
            
            prediction[prediction >= threshold] += 0.5
            prediction[prediction > 1] = 1
            prediction[prediction < threshold] = 0
            
            notes =  np.random.binomial(1, p=prediction)
            
        elif how == 'thresholded':
            
            prediction[prediction >= threshold] = 1
            prediction[prediction < threshold] = 0
            
            notes = prediction
        
        return notes
    

class Config(object):

    def __init__(self, d):

        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [Config(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, Config(b) if isinstance(b, dict) else b)
    
    @staticmethod
    def __read_config_file__(name):

        config_dict = {}
        with open(name) as f:
            data = json.load(f)
            for p in data:
                config_dict[p] = data[p]

        return config_dict


if __name__ == "__main__":

    file = 'maestro-v2.0.0/maestro-v2.0.0.csv'
    what_type = 'train'
    link = '2006/MIDI-Unprocessed_06_R1_2006_01-04_ORIG_MID--AUDIO_06_R1_2006_01_Track01_wav.midi'
    name = 'test2s21ded1dwsds32'
    contextLength = 100 # in timesteps
    start = 100
    length = 30
    numRegions = 2
    regionStarts = [110, 125]
    regionLengths = [8, 4]

    params = Config(Config.__read_config_file__('generation_config.json'))
    params = params.generation
    params.batch_size = 1

    curr_song = FakeSong(file = file,
                        what_type = what_type,
                        link = link,
                        name = name,
                        contextLength = contextLength,
                        start = start,
                        length = length,
                        numRegions = numRegions,
                        regionStarts = regionStarts,
                        regionLengths = regionLengths)

    curr_song.fill_gaps(params)

    





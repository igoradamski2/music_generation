from keras.models import Model
from keras.layers import Dense, Input, Lambda, Concatenate, LSTM, MaxPooling2D, Conv2D, Flatten

from keras import backend as K

import tensorflow as tf
#import tensorflow_probability as tfp # for tf version 2.0.0, tfp version 0.8 is needed 


def biaxial_target_model_oneseq(training_batch, encoder_output_size = 10):

    context_shape = training_batch.context.shape # [num_context,batch_size,note_size]
    target_shape  = training_batch.target_train.shape  # [batch_size, timesteps, note_size, note_features]

    # ----------------- here define model (128, 16, 78, 82)

    input_context = Input(batch_shape = 
                          (context_shape[0],  # batch_size
                           context_shape[1],  # num_of_contexts
                           context_shape[2],  # timesteps
                           context_shape[3]), # note_size
                          name="Input_layer_context") # as above
    
    input_target  = Input(batch_shape = 
                          (target_shape[0],  # batch_size
                           target_shape[1],  # timesteps
                           target_shape[2],  # note_size
                           target_shape[3]), # note_features
                          name="Input_layer_target")


    encoder = input_context
    encoder = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
                                      name="Encoder_layer_1")(encoder)
    encoder = LSTM(units = 512, 
                   dropout = 0.25, 
                   name = 'Encoder_lstm_1', 
                   return_sequences = False)(encoder)
    encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_1')(encoder)
    encoder = Dense(encoder_output_size, activation = 'softmax', name = "Encoder_output")(encoder)

    #encoder = Lambda(lambda x: K.mean(tf.reshape(x, 
                                      #[context_shape[0], 
                                       #context_shape[1], 
                                       #encoder_output_size]),
                               #axis = 0),
                    #name = "Encoder_mean_representation"
    #)
    encoder = Lambda(lambda x: tf.concat([

    tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                encoder_output_size])[:,0,:],
    tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                encoder_output_size])[:,1,:]],
                            axis = -1),
                    name = "Encoder_concat_representation"
    )(encoder)

    # Decoder
    propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, 1), 1), [1, target_shape[1], target_shape[2], 1]),
                               name = "Encoder_output_reshape")(encoder)
    propagate_in_time = Lambda(lambda x: tf.concat([input_target, x], axis = -1),
                               name = "Decoder_layer_1")(propagate_in_time)

    # TIME AXIS
    decoder = Lambda(lambda x: tf.reshape(tf.transpose(x, perm = [0,2,1,3]), 
                                          [-1,target_shape[1],target_shape[3]+2*encoder_output_size]))(propagate_in_time)
    decoder = LSTM(units = 200,
                   dropout = 0.25, 
                   name = "Decoder_time_lstm_1",
                   return_sequences = True)(decoder)
    decoder = LSTM(units = 200, 
                   dropout = 0.25,
                   name = "Decoder_time_lstm_2",
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(x, [-1, target_shape[2], 200]))(decoder)


    # NOTE AXIS
    decoder = LSTM(units = 100,
                   dropout = 0.25,
                   name = "Decoder_note_lstm_1",
                   return_sequences = True)(decoder)

    decoder = LSTM(units = 1,
                   dropout = 0.25,
                   name = "Decoder_note_lstm_2",
                   activation = 'sigmoid',
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(tf.squeeze(x), [target_shape[0], target_shape[1], target_shape[2]]))(decoder)
    decoder = Lambda(lambda x: x[:,-1,:])(decoder)

    model = Model([input_context, input_target], decoder)

    return model

def biaxial_target_model_meanrep(training_batch, encoder_output_size = 10):

    context_shape = training_batch.context.shape # [num_context,batch_size,note_size]
    target_shape  = training_batch.target_train.shape  # [batch_size, timesteps, note_size, note_features]

    # ----------------- here define model (128, 16, 78, 82)

    input_context = Input(batch_shape = 
                          (context_shape[0],  # batch_size
                           context_shape[1],  # num_of_contexts
                           context_shape[2],  # timesteps
                           context_shape[3]), # note_size
                          name="Input_layer_context") # as above
    
    input_target  = Input(batch_shape = 
                          (target_shape[0],  # batch_size
                           target_shape[1],  # timesteps
                           target_shape[2],  # note_size
                           target_shape[3]), # note_features
                          name="Input_layer_target")

    encoder = input_context
    encoder = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
                                      name="Encoder_layer_1")(encoder)
    encoder = LSTM(units = 512, 
                   dropout = 0.25, 
                   name = 'Encoder_lstm_1', 
                   return_sequences = False)(encoder)
    encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_1')(encoder)
    encoder = Dense(encoder_output_size, activation = 'softmax', name = "Encoder_output")(encoder)

    encoder = Lambda(lambda x: K.mean(tf.reshape(x, 
                                      [context_shape[0], 
                                       context_shape[1], 
                                       encoder_output_size]),
                               axis = 1),
                    name = "Encoder_mean_representation",
    )(encoder)
    #encoder = Lambda(lambda x: tf.concat([

    #tf.reshape(x, 
    #                           [context_shape[0], 
    #                            context_shape[1], 
    #                            encoder_output_size])[:,0,:],
    #tf.reshape(x, 
    #                           [context_shape[0], 
    #                            context_shape[1], 
    #                            encoder_output_size])[:,1,:]],
    #                        axis = -1),
    #                name = "Encoder_concat_representation"
    #)(encoder)

    # Decoder
    propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, 1), 1), [1, target_shape[1], target_shape[2], 1]),
                               name = "Encoder_output_reshape")(encoder)
    propagate_in_time = Lambda(lambda x: tf.concat([input_target, x], axis = -1),
                               name = "Decoder_layer_1")(propagate_in_time)

    # TIME AXIS
    decoder = Lambda(lambda x: tf.reshape(tf.transpose(x, perm = [0,2,1,3]), 
                                          [-1,target_shape[1],target_shape[3]+encoder_output_size]))(propagate_in_time)
    decoder = LSTM(units = 200,
                   dropout = 0.2, 
                   name = "Decoder_time_lstm_1",
                   return_sequences = True)(decoder)
    decoder = LSTM(units = 200, 
                   dropout = 0.2,
                   name = "Decoder_time_lstm_2",
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(x, [-1, target_shape[2], 200]))(decoder)


    # NOTE AXIS
    decoder = LSTM(units = 100,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_1",
                   return_sequences = True)(decoder)

    decoder = LSTM(units = 1,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_2",
                   activation = 'sigmoid',
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(tf.squeeze(x), [target_shape[0], target_shape[1], target_shape[2]]))(decoder)

    model = Model([input_context, input_target], decoder)

    return model

def biaxial_pn_encoder_concat_conv2d(training_batch, encoder_output_size = 10):

    context_shape = training_batch.context.shape # [num_context,batch_size,note_size]
    target_shape  = training_batch.target_train.shape  # [batch_size, timesteps, note_size, note_features]

    # ----------------- here define model (128, 16, 78, 82)

    input_context = Input(batch_shape = 
                          (context_shape[0],  # batch_size
                           context_shape[1],  # num_of_contexts
                           context_shape[2],  # timesteps
                           context_shape[3]), # note_size
                          name="Input_layer_context") # as above
    
    input_target  = Input(batch_shape = 
                          (target_shape[0],  # batch_size
                           target_shape[1],  # timesteps
                           target_shape[2],  # note_size
                           target_shape[3]), # note_features
                          name="Input_layer_target")

    encoder = input_context
    encoder = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
                                      name="Encoder_layer_1")(encoder)
    encoder = Lambda(lambda x: tf.expand_dims(x, -1))(encoder)

    encoder = Conv2D(filters = 64, 
                     kernel_size = (int(context_shape[2]/5), int(context_shape[3]/7)), 
                     name = 'Encoder_conv_1')(encoder)
    encoder = MaxPooling2D(pool_size=(3, 3))(encoder)
    encoder = Conv2D(filters = 64, 
                     kernel_size = (int(context_shape[2]/5), int(context_shape[3]/7)), 
                     name = 'Encoder_conv_2')(encoder) 
    encoder = Flatten()(encoder) 
    encoder = Dense(256, activation = 'relu', name = 'Encoder_dense_1')(encoder)
    encoder = Dense(78, activation = 'softmax', name = "Encoder_output")(encoder)
    #encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_1')(encoder)
    #encoder = Dense(78, activation = 'softmax', name = "Encoder_output")(encoder)

    #encoder = Lambda(lambda x: K.mean(tf.reshape(x, 
                                      #[context_shape[0], 
                                       #context_shape[1], 
                                       #encoder_output_size]),
                               #axis = 0),
                    #name = "Encoder_mean_representation"
    #)
    encoder = Lambda(lambda x: tf.concat([

    tf.expand_dims(
        tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                context_shape[3]])[:,0,:],
        -1),
    tf.expand_dims(
        tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                context_shape[3]])[:,1,:],
        -1)
    ],
                            axis = -1),
                    name = "Encoder_concat_representation"
    )(encoder)

    # Decoder
    propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, target_shape[1], 1, 1]),
                               name = "Encoder_output_reshape")(encoder)
    propagate_in_time = Lambda(lambda x: tf.concat([input_target, x], axis = -1),
                               name = "Decoder_layer_1")(propagate_in_time)

    # TIME AXIS
    decoder = Lambda(lambda x: tf.reshape(tf.transpose(x, perm = [0,2,1,3]), 
                                          [-1,target_shape[1],target_shape[3]+2]))(propagate_in_time)
    decoder = LSTM(units = 200,
                   dropout = 0.2, 
                   name = "Decoder_time_lstm_1",
                   return_sequences = True)(decoder)
    decoder = LSTM(units = 200, 
                   dropout = 0.2,
                   name = "Decoder_time_lstm_2",
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(x, [-1, target_shape[2], 200]))(decoder)


    # NOTE AXIS
    decoder = LSTM(units = 100,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_1",
                   return_sequences = True)(decoder)

    decoder = LSTM(units = 1,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_2",
                   activation = 'sigmoid',
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(tf.squeeze(x), [target_shape[0], target_shape[1], target_shape[2]]))(decoder)

    model = Model([input_context, input_target], decoder)

    return model

def biaxial_target_model(training_batch, encoder_output_size = 10):

    context_shape = training_batch.context.shape # [num_context,batch_size,note_size]
    target_shape  = training_batch.target_train.shape  # [batch_size, timesteps, note_size, note_features]

    # ----------------- here define model (128, 16, 78, 82)

    input_context = Input(batch_shape = 
                          (context_shape[0],  # batch_size
                           context_shape[1],  # num_of_contexts
                           context_shape[2],  # timesteps
                           context_shape[3]), # note_size
                          name="Input_layer_context") # as above
    
    input_target  = Input(batch_shape = 
                          (target_shape[0],  # batch_size
                           target_shape[1],  # timesteps
                           target_shape[2],  # note_size
                           target_shape[3]), # note_features
                          name="Input_layer_target")

    encoder = input_context
    encoder = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
                                      name="Encoder_layer_1")(encoder)
    encoder = LSTM(units = 512, 
                   dropout = 0.25, 
                   name = 'Encoder_lstm_1', 
                   return_sequences = False)(encoder)
    encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_1')(encoder)
    encoder = Dense(encoder_output_size, activation = 'softmax', name = "Encoder_output")(encoder)

    #encoder = Lambda(lambda x: K.mean(tf.reshape(x, 
                                      #[context_shape[0], 
                                       #context_shape[1], 
                                       #encoder_output_size]),
                               #axis = 0),
                    #name = "Encoder_mean_representation"
    #)
    encoder = Lambda(lambda x: tf.concat([

    tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                encoder_output_size])[:,0,:],
    tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                encoder_output_size])[:,1,:]],
                            axis = -1),
                    name = "Encoder_concat_representation"
    )(encoder)

    # Decoder
    propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, 1), 1), [1, target_shape[1], target_shape[2], 1]),
                               name = "Encoder_output_reshape")(encoder)
    propagate_in_time = Lambda(lambda x: tf.concat([input_target, x], axis = -1),
                               name = "Decoder_layer_1")(propagate_in_time)

    # TIME AXIS
    decoder = Lambda(lambda x: tf.reshape(tf.transpose(x, perm = [0,2,1,3]), 
                                          [-1,target_shape[1],target_shape[3]+2*encoder_output_size]))(propagate_in_time)
    decoder = LSTM(units = 200,
                   dropout = 0.2, 
                   name = "Decoder_time_lstm_1",
                   return_sequences = True)(decoder)
    decoder = LSTM(units = 200, 
                   dropout = 0.2,
                   name = "Decoder_time_lstm_2",
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(x, [-1, target_shape[2], 200]))(decoder)


    # NOTE AXIS
    decoder = LSTM(units = 100,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_1",
                   return_sequences = True)(decoder)

    decoder = LSTM(units = 1,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_2",
                   activation = 'sigmoid',
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(tf.squeeze(x), [target_shape[0], target_shape[1], target_shape[2]]))(decoder)

    model = Model([input_context, input_target], decoder)

    return model

def biaxial_pn_encoder_concat_deeplstm(training_batch, encoder_output_size = 32):

    context_shape = training_batch.context.shape # [num_context,batch_size,note_size]
    target_shape  = training_batch.target_train.shape  # [batch_size, timesteps, note_size, note_features]

    # ----------------- here define model (128, 16, 78, 82)

    input_context = Input(batch_shape = 
                          (context_shape[0],  # batch_size
                           context_shape[1],  # num_of_contexts
                           context_shape[2],  # timesteps
                           context_shape[3]), # note_size
                          name="Input_layer_context") # as above
    
    input_target  = Input(batch_shape = 
                          (target_shape[0],  # batch_size
                           target_shape[1],  # timesteps
                           target_shape[2],  # note_size
                           target_shape[3]), # note_features
                          name="Input_layer_target")

    encoder = input_context
    encoder = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
                                      name="Encoder_layer_1")(encoder)
    encoder = Lambda(lambda x: tf.transpose(tf.expand_dims(x, -1), perm = [0,2,1,3]))(encoder)
    encoder = Lambda(lambda x: tf.reshape(x, [-1, x.shape[2], x.shape[3]]))(encoder)
    
    encoder = LSTM(units = 200,
                   dropout = 0.2, 
                   name = "Encoder_time_lstm_1",
                   return_sequences = True)(encoder)
    encoder = LSTM(units = 200, 
                   dropout = 0.2,
                   name = "Encoder_time_lstm_2",
                   return_sequences = False)(encoder)

    encoder = Lambda(lambda x: tf.reshape(x, [-1, target_shape[2], 200]))(encoder)


    # NOTE AXIS
    encoder = LSTM(units = 200,
                   dropout = 0.2,
                   name = "Encoder_note_lstm_1",
                   return_sequences = True)(encoder)
    
    encoder = LSTM(units = encoder_output_size,
                   dropout = 0.2,
                   name = "Encoder_note_lstm_2",
                   return_sequences = True,
                   activation = 'linear')(encoder)

    #encoder = Lambda(lambda x: tf.reshape(tf.squeeze(x), [target_shape[0], target_shape[1], target_shape[2]]))(encoder)
    
    
    #encoder = Lambda(lambda x: K.mean(tf.reshape(x, 
                                      #[context_shape[0], 
                                       #context_shape[1], 
                                       #encoder_output_size]),
                               #axis = 0),
                    #name = "Encoder_mean_representation"
    #)
    encoder = Lambda(lambda x: tf.concat([

        tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                context_shape[3],
                                encoder_output_size])[:,0,:,:],
        tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                context_shape[3],
                                encoder_output_size])[:,1,:,:]
    ],
                            axis = -1),
                    name = "Encoder_concat_representation"
    )(encoder)

    # Decoder
    propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, target_shape[1], 1, 1]),
                               name = "Encoder_output_reshape")(encoder)
    propagate_in_time = Lambda(lambda x: tf.concat([input_target, x], axis = -1),
                               name = "Decoder_layer_1")(propagate_in_time)

    # TIME AXIS
    decoder = Lambda(lambda x: tf.reshape(tf.transpose(x, perm = [0,2,1,3]), 
                                          [-1,target_shape[1],target_shape[3]+2*encoder_output_size]))(propagate_in_time)
    decoder = LSTM(units = 200,
                   dropout = 0.2, 
                   name = "Decoder_time_lstm_1",
                   return_sequences = True)(decoder)
    decoder = LSTM(units = 200, 
                   dropout = 0.2,
                   name = "Decoder_time_lstm_2",
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(x, [-1, target_shape[2], 200]))(decoder)


    # NOTE AXIS
    decoder = LSTM(units = 100,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_1",
                   return_sequences = True)(decoder)

    decoder = LSTM(units = 1,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_2",
                   activation = 'sigmoid',
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(tf.squeeze(x), [target_shape[0], target_shape[1], target_shape[2]]))(decoder)

    model = Model([input_context, input_target], decoder)

    return model

def biaxial_target_conv2d_model(training_batch, encoder_output_size = 78):

    context_shape = training_batch.context.shape # [num_context,batch_size,note_size]
    target_shape  = training_batch.target_train.shape  # [batch_size, timesteps, note_size, note_features]

    # ----------------- here define model (128, 16, 78, 82)

    input_context = Input(batch_shape = 
                          (context_shape[0],  # batch_size
                           context_shape[1],  # num_of_contexts
                           context_shape[2],  # timesteps
                           context_shape[3]), # note_size
                          name="Input_layer_context") # as above
    
    input_target  = Input(batch_shape = 
                          (target_shape[0],  # batch_size
                           target_shape[1],  # timesteps
                           target_shape[2],  # note_size
                           target_shape[3]), # note_features
                          name="Input_layer_target")

    encoder = input_context
    encoder = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
                                      name="Encoder_layer_1")(encoder)
    encoder = Lambda(lambda x: tf.expand_dims(x, -1))(encoder)

    encoder = Conv2D(filters = 64, 
                     kernel_size = (int(context_shape[2]/5), int(context_shape[3]/7)), 
                     name = 'Encoder_conv_1')(encoder)
    encoder = MaxPooling2D(pool_size=(3, 3))(encoder)
    encoder = Conv2D(filters = 64, 
                     kernel_size = (int(context_shape[2]/5), int(context_shape[3]/7)), 
                     name = 'Encoder_conv_2')(encoder) 
    encoder = Flatten()(encoder) 
    encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_1')(encoder)
    encoder = Dense(encoder_output_size, activation = 'softmax', name = "Encoder_output")(encoder)

    #encoder = Lambda(lambda x: K.mean(tf.reshape(x, 
                                      #[context_shape[0], 
                                       #context_shape[1], 
                                       #encoder_output_size]),
                               #axis = 0),
                    #name = "Encoder_mean_representation"
    #)
    encoder = Lambda(lambda x: tf.concat([

    tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                encoder_output_size])[:,0,:],
    tf.reshape(x, 
                               [context_shape[0], 
                                context_shape[1], 
                                encoder_output_size])[:,1,:]],
                            axis = -1),
                    name = "Encoder_concat_representation"
    )(encoder)

    # Decoder
    propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(tf.expand_dims(x, 1), 1), [1, target_shape[1], target_shape[2], 1]),
                               name = "Encoder_output_reshape")(encoder)
    propagate_in_time = Lambda(lambda x: tf.concat([input_target, x], axis = -1),
                               name = "Decoder_layer_1")(propagate_in_time)

    # TIME AXIS
    decoder = Lambda(lambda x: tf.reshape(tf.transpose(x, perm = [0,2,1,3]), 
                                          [-1,target_shape[1],target_shape[3]+2*encoder_output_size]))(propagate_in_time)
    decoder = LSTM(units = 200,
                   dropout = 0.2, 
                   name = "Decoder_time_lstm_1",
                   return_sequences = True)(decoder)
    decoder = LSTM(units = 200, 
                   dropout = 0.2,
                   name = "Decoder_time_lstm_2",
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(x, [-1, target_shape[2], 200]))(decoder)


    # NOTE AXIS
    decoder = LSTM(units = 100,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_1",
                   return_sequences = True)(decoder)

    decoder = LSTM(units = 1,
                   dropout = 0.2,
                   name = "Decoder_note_lstm_2",
                   activation = 'sigmoid',
                   return_sequences = True)(decoder)

    decoder = Lambda(lambda x: tf.reshape(tf.squeeze(x), [target_shape[0], target_shape[1], target_shape[2]]))(decoder)

    model = Model([input_context, input_target], decoder)

    return model


def simple_model(training_batch,
    lstm_units = 512,
    embedding_size = 128,
    encoder_dropout = 0.1,
    decoder_dropout = 0.1):

    context_shape = training_batch.context.shape # [batch_size,num_windows,timesteps_per_window,note_size]
    target_shape  = training_batch.target.shape  # [batch_size, timesteps, note_size]

    # ----------------- here define model

    input_context = Input(batch_shape = 
                          (context_shape[0],  # context_num
                           context_shape[1],  # batch_size
                           context_shape[2],  # timesteps
                           context_shape[3]), # note_size
                          name="Input_layer_context") # as above
    
    input_target  = Input(batch_shape = 
                          (target_shape[0],  # batch_size
                           target_shape[1],  # timesteps
                           target_shape[2]), # note_size
                          name="Input_layer_target")  


    # Encoder

    encoder = input_context
    
    reshape_input_to_windows = Lambda(lambda x: tf.reshape(x, [-1,x.shape[2],x.shape[3]]), 
                                      name="Reshape_layer_1")(encoder)
    
    encoder = LSTM(units = lstm_units, 
                   dropout = encoder_dropout, 
                   name = 'Encoder_lstm_1', 
                   return_sequences = True)(reshape_input_to_windows)
    
    encoder = LSTM(units = lstm_units, 
                   dropout = encoder_dropout, 
                   name = 'Encoder_lstm_2')(encoder)

    encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_1')(encoder)
    #encoder = Dense(512, activation = 'relu', name = 'Encoder_dense_2')(encoder)
    encoder = Dense(embedding_size, activation = 'tanh', name = 'Encoder_dense_3')(encoder)

    mean_representation = Lambda(lambda x: K.mean(tf.reshape(x, [context_shape[0], 
                                                                 context_shape[1], 
                                                                 embedding_size]), axis = -2),
                                 name="Mean_representation_layer")(encoder)

    # Decoder

    propagate_in_time = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1,target_shape[1],1]),
                               name = "Encoder_lambda_1")(mean_representation)

    decoder_input = Lambda(lambda x: tf.concat([input_target, x], axis = 2),
                           name = "Encoder_lambda_2")(propagate_in_time)
    #decoder_input = propagate_in_time

    decoder, _, _ = LSTM(units = 512, 
                      dropout = decoder_dropout,
                      return_sequences = True,
                      return_state = True,
                      activation = 'tanh',
                      name = 'Decoder_lstm_1')(decoder_input)

    decoder, _, _ = LSTM(units = context_shape[3], 
                      dropout = decoder_dropout,
                      return_sequences = True,
                      return_state = True,
                      activation = 'sigmoid',
                      name = 'Decoder_lstm_2')(decoder)


    model = Model([input_context, input_target], decoder)

    return model

def get_decoder_simple(model):

    input_shape = model.get_layer("lambda_2").output.shape

    input_embedding = Input(batch_shape = 
                          (input_shape[0],  # batch_size
                           None,            # timesteps ()
                           input_shape[2]),  # note_size
                          name="Input_layer_embedding")

    decoder, _, _ = LSTM(units = 512, 
                      return_sequences = True,
                      return_state = True,
                      activation = 'tanh',
                      name = 'Decoder_lstm_1')(input_embedding)

    decoder = LSTM(units = 88, 
                      activation = 'sigmoid',
                      name = 'Decoder_lstm_2')(decoder)

    new_model = Model(input_embedding, decoder)

    names = {layer.name:idx for idx, layer in enumerate(model.layers)}
    weights = model.get_weights()

    for idx, layer in model.layers:
        if layer.name in names.keys():
            new_model.layers[idx] = weights[names[layer.name]]

    return model

def get_encoder_simple(model):

    input_context = model.get_layer("Input_layer_context")

    encoder = model.get_layer("Reshape_layer_1")(input_context)
    encoder = model.get_layer("Encoder_lstm_1")(encoder)
    encoder = model.get_layer("Encoder_lstm_2")(encoder)
    encoder = model.get_layer("Encoder_dense_1")(encoder)
    encoder = model.get_layer("Encoder_dense_3")(encoder)
    encoder = model.get_layer("Mean_representation_layer")(encoder)

    model = Model(input_context, encoder)

    return model

















import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Maximum, Average, Minimum, Dense, TimeDistributed, LSTM, Dropout
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.utils import plot_model
from tensorflow.keras.backend import stack
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

import keras.backend as K


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class baseline_model(object):
    def __init__(self, img_num=8):
        self.frame_shape = 1000
        self.info_shape = 1000
    def construct(self):
        frame_shape = self.frame_shape
        info_shape = self.info_shape
        input_tensors = []
        frame_input = Input(shape=frame_shape)
        info_input = Input(shape=info_shape)
        input_tensors.append(frame_input)
        input_tensors.append(info_input)


#        concate_output = concatenate((tf.cast(tf.convert_to_tensor(frame_input), 'float32'), mean_background), axis=-1)

        dense_output_1 = Dense(512, activation="relu", name="dense_output_1")(frame_input)
  
        dense_output_2 = Dense(512, activation="relu", name="dense_output_2")(info_input)
        concate_output = concatenate((dense_output_1, dense_output_2), axis=-1)
        dense_output_3 = Dense(512, activation="relu", name="dense_output_3")(concate_output)
        dense_output_4 = Dense(128, activation="relu", name="dense_output_4")(dense_output_3)
        position_output = Dense(1, activation="sigmoid", name="position")(dense_output_4)


        output_tensors = []
        output_tensors.append(position_output)
        fr_model = Model(input_tensors, output_tensors)

        fr_model.summary()
        #        config = avc.get_config()
        #        new_model = keras.Model.from_config(config)
        return fr_model


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
        self.frame_shape = (1000)
        self.info_shape = (1000)
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
    
def train_single_w_rank(data_path: str, initial_weights_path: str, save_weights_path: str, batch_size: int):
    #batch_size = 16
    epochs_1st = 200

    gpu_n = 1  # gpu number

    model_base = baseline_model()

    model_single = model_base.construct()
    frame_shape = (1000)
    info_shape = (1000)

    frame_input_l = Input(shape=frame_shape)
    info_input_l = Input(shape=info_shape)
    frame_input_r = Input(shape=frame_shape)
    info_input_r = Input(shape=info_shape)

    l_out = model_single([frame_input_l, info_input_l])
    r_out = model_single([frame_input_r, info_input_r])
    #diff = Subtract()([l_out, r_out])
    #prob = dense_diff(diff)
    #rank_model = Model([frame_input_l, frame_input_r], prob)
    diff = Subtract()([l_out, r_out])
    diff = diff * 0.5
    b = tf.constant(0.5, shape=(1,))
    prob = tf.keras.layers.Add()([diff, b])
    rank_model = Model([[frame_input_l, info_input_l], [frame_input_r, info_input_r]], prob)

    rank_model.summary()
    adamopt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

    rank_model.compile(optimizer=adamopt, loss=['BinaryCrossentropy'], metrics=["accuracy"])
    
if __name__ == "__main__":
    train_single_w_rank("a", "a", "a", 1)

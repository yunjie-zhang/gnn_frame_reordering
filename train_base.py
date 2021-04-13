import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Subtract, Input, Maximum, Average, Minimum, Dense, TimeDistributed, LSTM, Dropout
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
    dense_diff = Dense(1, activation="sigmoid", name="diff_output")
    #diff = Subtract()([l_out, r_out])
    #prob = dense_diff(diff)
    #rank_model = Model([frame_input_l, frame_input_r], prob)
    diff = Subtract()([l_out, r_out])
    #diff = diff * 0.5
    #b = tf.constant(0.5, shape=(1,))
    #prob = tf.keras.layers.Add()([diff, b])
    prob = dense_diff(diff)
    rank_model = Model([[frame_input_l, info_input_l], [frame_input_r, info_input_r]], prob)

    rank_model.summary()
    adamopt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

    rank_model.compile(optimizer=adamopt, loss=['BinaryCrossentropy'], metrics=["accuracy"])
    
    image_features_l_train = np.load("img_feature_left_train.npy")
    image_features_r_train = np.load("img_feature_right_train.npy")
    info_features_train = np.load("account_feature_train.npy")
    target_train = np.load("target_train.npy")
    
    train_index = int(len(target_train) * 0.95)
    
    image_features_l_dev = image_features_l_train[train_index:]
    image_features_r_dev = image_features_r_train[train_index:]
    info_features_dev = info_features_train[train_index:]
    target_dev = target_train[train_index:]
    
    image_features_l_train = image_features_l_train[0: train_index]
    image_features_r_train = image_features_r_train[0: train_index]
    info_features_train = info_features_train[0: train_index]
    target_train = target_train[0: train_index]
    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    rank_model.fit([[image_features_l_train, info_features_train],[image_features_r_train, info_features_train]], 
                   target_train, verbose=1, batch_size=16, epochs=50, callbacks=[early_stopping], 
                   validation_data=([[image_features_l_dev, info_features_dev],[image_features_r_dev, info_features_dev]], target_dev))
    
    image_features_l_test = np.load("img_feature_left_test.npy")
    image_features_r_test = np.load("img_feature_right_test.npy")
    info_features_test = np.load("account_feature_test.npy")
    target_test = np.load("target_test.npy")
    results = rank_model.evaluate([[image_features_l_test, info_features_test],[image_features_r_test, info_features_test]], target_test, batch_size=128)
    print("Result: ", results)
    
if __name__ == "__main__":
    train_single_w_rank("a", "a", "a", 1)

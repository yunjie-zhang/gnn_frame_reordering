import numpy as np
#from musicnn.extractor import extractor
import re
from multiprocessing import Pool
import csv
import tensorflow as tf
import os,shutil, sys
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from PIL import ImageFile
import pickle
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import random
import cv2
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True
image_shape = (224, 224, 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

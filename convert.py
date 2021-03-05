# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:04:26 2020

@author: THUAN
"""

import tensorflow as tf

TF_LITE_MODEL_FILE_NAME = "E:/Python/trafficsignsCNN/tf_lite_model.tflite"

keras_model = tf.keras.models.load_model("E:/Python/trafficsignsCNN/tensorflow2.h5")
tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = tf_lite_converter.convert()

tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name, "wb").write(tflite_model)


convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME), "KB")

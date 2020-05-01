# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:50:08 2020

@author: nisha
"""
from tensorflow.keras.models import load_model
import tensorflow as tf


model = load_model('final_model_second.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)


tflite_model = converter.convert()

file = open( 'final_model_second.tflite' , 'wb' ).write( tflite_model )
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 10:05:00 2020

@author: nisha
"""
from tensorflow.keras.models import load_model
import tensorflow as tf

new_model= tf.keras.models.load_model(filepath=".h5")
input1 = new_model.input_shape()
output = new_model.output_shape()
print(input1)
print(output)
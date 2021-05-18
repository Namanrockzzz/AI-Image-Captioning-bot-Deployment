#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import json
import pickle
import tensorflow.keras.applications.vgg16 as VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.layers import add

model = load_model("model_19.h5")
model.make_predict_function()

model_temp = ResNet50(weights='imagenet',input_shape=(224,224,3))

model_resnet = Model(model_temp.input,model_temp.layers[-2].output) 
model_resnet.make_predict_function() 

def preprcess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img

def encode_img(img):
    img = preprcess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector.reshape((-1,))
    # print(feature_vector.shape)
    return feature_vector

with open("./storage/word_2_idx.pkl","rb") as f:
    word_2_idx = pickle.load(f)
with open("./storage/idx_2_word.pkl","rb") as f:
    idx_2_word = pickle.load(f)

def predict_caption(photo):
    in_text = "startseq"
    max_len = 35
    for i in range(max_len):
        sequence = [word_2_idx[w] for w in in_text.split() if w in word_2_idx]
        sequence = pad_sequences([sequence,],maxlen=max_len,padding="post")
        
        y_pred = model.predict([photo,sequence])
        y_pred = y_pred.argmax() # Word with maximum probab - Greedy Sampling
        word = idx_2_word[y_pred]
        in_text += (' '+ word)
        
        if word=="endseq":
            break
        
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption

def caption_this_image(img):
    enc = encode_img(img)
    caption = predict_caption(enc)
    return caption





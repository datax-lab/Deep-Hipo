import numpy as np

from keras.layers import Input,GlobalAveragePooling2D
import os
from helper import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, Dilationlayer

from keras.layers import  BatchNormalization,Activation,concatenate
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
from keras.models import Model,Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
        
class CATNet():

    def __init__(self, name):
        self.name = name

        
        
    def bulid(self,input_img):
        model  = Dilationlayer(self.name+"baselayer").bulid(input_img)
        model  = InceptionA(self.name+"mixedA1").bulid(model)
        model  = InceptionA(self.name+"mixedA2").bulid(model)
        model  = InceptionA(self.name+"mixedA3").bulid(model)
        model  = InceptionB(self.name+"mixedB1").bulid(model)
        model  = InceptionC(self.name+"mixedC1",128).bulid(model)
        model  = InceptionC(self.name+"mixedC2",160).bulid(model)
        model  = InceptionC(self.name+"mixedC3",160).bulid(model)
        model  = InceptionC(self.name+"mixedC4",192).bulid(model)
        model  = InceptionD(self.name+"mixedD1").bulid(model)
        model  = InceptionE(self.name+"mixedE1").bulid(model)
        model  = InceptionE(self.name+"mixedE2").bulid(model)
        model = Model(input_img,model)
        return model


        

class CATNet2():
    
    def __init__(self, name):
        
        self.name = name

        

    def bulid(InputA,InputB,modelname = "CAT_Net2"):
        

        CATNet_Track1 = CATNet(modelname +"Track1").bulid(InputA)
        CATNet_Track1output = CATNet_Track1.output
        CATNet_Track2 = CATNet(modelname+"Track2").bulid(InputB)
        CATNet_Track2output = CATNet_Track2.output
        CATNet_Track1flat = GlobalAveragePooling2D()(CATNet_Track1output)
        CATNet_Track2flat = GlobalAveragePooling2D()(CATNet_Track2output)
        combined = concatenate([CATNet_Track1flat, CATNet_Track2flat])
        z = Dense(2048, activation='relu')(combined)
        predictions = Dense(1, activation='softmax')(z)
        model = Model(inputs=[CATNet_Track1.input,CATNet_Track2.input], outputs=predictions)
        return model

    

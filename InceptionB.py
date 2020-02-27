from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import keras
from keras import layers
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D, AveragePooling2D
from keras.layers import  BatchNormalization,Activation,concatenate,Dropout,SpatialDropout2D
from keras.models import Model,Sequential
from keras.regularizers import l2
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# #The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
        
class IndiviualCNN():

    def __init__(self, filters,num_row, num_col,name,strides = (1,1)):
        self.strides = strides
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.name = name
        self.conv_name = self.name+'_conv'
        self.bn_name = self.name+'_bn'
        self.Padding = "same"
        self.dilation_rate = (1,1)
        self.eps = 0.001
        self.conv_name = self.name+'_conv'
        self.Kernel_regularizer = l2(0.02)
        
    def bulid(self,inputlayer):
    
        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,strides=self.strides,padding=self.Padding,use_bias=False,name = self.conv_name, kernel_regularizer = self.Kernel_regularizer)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= False,  name = self.bn_name)(Dilatedlayer)
        # BatchNor = Dropout(0.2)(BatchNor)
        BatchNor = SpatialDropout2D(0.4)(BatchNor)
        Act = Activation('tanh', name= self.name)(BatchNor)
        return Act
        

class InceptionB():
    
    def __init__(self, name):
        
        self.name = name

        

    def bulid(self,inputmodel):
        #branch1
        branch3x3 = IndiviualCNN(384,3,3,name = self.name+"inceptionb3x3",strides = (2,2)).bulid(inputmodel)

        #branch2
        branch3x3_1 = IndiviualCNN(64,1,1,name = self.name+"inceptionb3x3_1").bulid(inputmodel)
        branch3x3_2 = IndiviualCNN(96,3,3,name = self.name+"inceptionb3x3_2").bulid(branch3x3_1)
        branch3x3_3 = IndiviualCNN(96,3,3,name = self.name+"inceptionb3x3_3",strides = (2,2)).bulid(branch3x3_2)

        #branch3
        branchpool = MaxPooling2D((3, 3),strides=(2, 2),padding='same')(inputmodel)
        output = [branch3x3,branch3x3_3,branchpool]
        #concancate
        IncB = concatenate(output,axis=3,name = self.name)
        return IncB

     

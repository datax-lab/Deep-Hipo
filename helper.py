from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers import  BatchNormalization,Activation

class IndiviualCNN():

    def __init__(self, filters,num_row, num_col,name,strides = (1,1), dilation_rate = (1,1)):
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.name = name
        self.conv_name = self.name+'_conv'
        self.bn_name = self.name+'_bn'
        self.Padding = "valid"
        self.dilation_rate = dilation_rate
        self.eps = 0.001
        self.conv_name = self.name+'_conv'
        self.strides = strides


    def bulid(self,inputlayer):

        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,strides=self.strides,padding=self.Padding,use_bias=False,name = self.conv_name)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= True, epsilon = self.eps, name = self.bn_name)(Dilatedlayer)
        Act = Activation('relu', name= self.name)(BatchNor)
        return Act
        

class Dilationlayer():
    
    def __init__(self, name):
        
        self.name = name


    def bulid(self,inputmodel):
        Dil1 = IndiviualCNN(32,3,3,name = self.name+"dil1",dilation_rate = (2,2)).bulid(inputmodel)
        Dil2 = IndiviualCNN(32,3,3,name = self.name+"dil2",dilation_rate = (2,2)).bulid(Dil1)
        Dil3 = IndiviualCNN(64,3,3,name = self.name+"dil3",dilation_rate = (2,2)).bulid(Dil2)
        Dil4 = IndiviualCNN(64,1,1,name = self.name+"dil4",dilation_rate = (2,2)).bulid(Dil3)
        Dil5 = IndiviualCNN(80,3,3,name = self.name+"dil5",dilation_rate = (2,2)).bulid(Dil4)
        return Dil5


class InceptionA():

    def __init__(self, name):
        self.name = name

    def bulid(self, inputmodel):
        # branch1
        branch1x1 = IndiviualCNN(64, 1, 1, name=self.name + "inceptiona1x1").bulid(inputmodel)

        # branch2
        branch5x5_1 = IndiviualCNN(48, 1, 1, name=self.name + "inceptiona5x5_1").bulid(inputmodel)
        branch5x5_2 = IndiviualCNN(64, 5, 5, name=self.name + "inceptiona5x5_2").bulid(branch5x5_1)

        # branch3
        branch3x3_1 = IndiviualCNN(64, 1, 1, name=self.name + "inceptiona3x3_1").bulid(inputmodel)
        branch3x3_2 = IndiviualCNN(96, 3, 3, name=self.name + "inceptiona3x3_2").bulid(branch3x3_1)
        branch3x3_3 = IndiviualCNN(96, 3, 3, name=self.name + "inceptiona3x3_3").bulid(branch3x3_2)

        # branch4
        branchpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputmodel)
        branchpool_1 = IndiviualCNN(32, 1, 1, name=self.name + "branchpool1").bulid(branchpool)

        output = [branch5x5_2, branch5x5_2, branch3x3_3, branchpool_1]
        # concancate
        IncA = concatenate(output, axis=3, name=self.name)
        return IncA


class InceptionB():

    def __init__(self, name):
        self.name = name

    def bulid(self, inputmodel):
        # branch1
        branch3x3 = IndiviualCNN(384, 3, 3, name=self.name + "inceptionb3x3", strides=(2, 2)).bulid(inputmodel)

        # branch2
        branch3x3_1 = IndiviualCNN(64, 1, 1, name=self.name + "inceptionb3x3_1").bulid(inputmodel)
        branch3x3_2 = IndiviualCNN(96, 3, 3, name=self.name + "inceptionb3x3_2").bulid(branch3x3_1)
        branch3x3_3 = IndiviualCNN(96, 3, 3, name=self.name + "inceptionb3x3_3", strides=(2, 2)).bulid(branch3x3_2)

        # branch3
        branchpool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputmodel)
        output = [branch3x3, branch3x3_3, branchpool]
        # concancate
        IncB = concatenate(output, axis=3, name=self.name)
        return IncB


class InceptionC():

    def __init__(self, name, c7):
        self.name = name
        self.c7 = c7

    def bulid(self, inputmodel):
        # branch1
        branch1x1 = IndiviualCNN(192, 1, 1, name=self.name + "inceptionc1x1").bulid(inputmodel)

        branch7x7_1 = IndiviualCNN(self.c7, 1, 1, name=self.name + "inceptionc7x7_1").bulid(inputmodel)
        branch7x7_2 = IndiviualCNN(self.c7, 1, 7, name=self.name + "inceptionc7x7_2").bulid(branch7x7_1)
        branch7x7_3 = IndiviualCNN(192, 7, 1, name=self.name + "inceptionc7x7_3").bulid(branch7x7_2)

        branch7x7_b1 = IndiviualCNN(self.c7, 1, 1, name=self.name + "inceptionc7x7c_1").bulid(inputmodel)
        branch7x7_b2 = IndiviualCNN(self.c7, 7, 1, name=self.name + "inceptionc7x7c_2").bulid(branch7x7_b1)
        branch7x7_b3 = IndiviualCNN(self.c7, 1, 7, name=self.name + "inceptionc7x7c_3").bulid(branch7x7_b2)
        branch7x7_b4 = IndiviualCNN(self.c7, 7, 1, name=self.name + "inceptionc7x7c_4").bulid(branch7x7_b3)
        branch7x7_b5 = IndiviualCNN(192, 1, 7, name=self.name + "inceptionc7x7c_5").bulid(branch7x7_b4)

        branchpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputmodel)
        branchpool_1 = IndiviualCNN(192, 1, 1, name=self.name + "branchpool1").bulid(branchpool)

        output = [branch1x1, branch7x7_3, branch7x7_b5, branchpool_1]

        IncB = concatenate(output, axis=3, name=self.name)
        return IncB


class InceptionD():

    def __init__(self, name):
        self.name = name

    def bulid(self, inputmodel):
        # branch1
        branch3x3_1 = IndiviualCNN(192, 1, 1, name=self.name + "inceptiond3x3").bulid(inputmodel)
        branch3x3_2 = IndiviualCNN(320, 3, 3, name=self.name + "inceptiond13x3", strides=(2, 2)).bulid(branch3x3_1)

        # branch2
        branch7x7_1 = IndiviualCNN(192, 1, 1, name=self.name + "inceptiond7x7_1").bulid(inputmodel)
        branch7x7_2 = IndiviualCNN(192, 1, 7, name=self.name + "inceptiond7x7_2").bulid(branch7x7_1)
        branch7x7_3 = IndiviualCNN(192, 7, 1, name=self.name + "inceptiond7x7_3").bulid(branch7x7_2)
        branch7x7_4 = IndiviualCNN(192, 3, 3, name=self.name + "inceptiond7x7_4", strides=(2, 2)).bulid(branch7x7_3)

        # branch3
        branchpool = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputmodel)

        output = [branch3x3_2, branch7x7_4, branchpool]
        # concancate
        IncB = concatenate(output, axis=3, name=self.name)
        return IncB

class InceptionE():

    def __init__(self, name):
        self.name = name

    def bulid(self, inputmodel):
        # branch1
        branch1x1_1 = IndiviualCNN(320, 1, 1, name=self.name + "inceptione1x1").bulid(inputmodel)

        # branch2
        branch3x3_1 = IndiviualCNN(384, 1, 1, name=self.name + "inceptione3x3_1").bulid(inputmodel)
        branch3x3_2a = IndiviualCNN(384, 1, 3, name=self.name + "inceptione3x3_2a").bulid(branch3x3_1)
        branch3x3_2b = IndiviualCNN(384, 3, 1, name=self.name + "inceptione3x3_2b").bulid(branch3x3_1)
        branch3x3 = concatenate([branch3x3_2a, branch3x3_2b], axis=3, name=self.name + "inceptione3x3_concat1")

        branch3x3_2 = IndiviualCNN(448, 1, 1, name=self.name + "inceptione3x3_2").bulid(inputmodel)
        branch3x3_3 = IndiviualCNN(384, 1, 1, name=self.name + "inceptione3x3_3").bulid(branch3x3_2)
        branch3x3_4a = IndiviualCNN(384, 1, 3, name=self.name + "inceptione3x3_4a").bulid(branch3x3_3)
        branch3x3_4b = IndiviualCNN(384, 3, 1, name=self.name + "inceptione3x3_4b").bulid(branch3x3_3)
        branch3x3_5 = concatenate([branch3x3_4a, branch3x3_4b], axis=3, name=self.name + "inceptione3x3_concat2")

        # branch3
        branchpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputmodel)
        branchpool_1 = IndiviualCNN(192, 1, 1, name=self.name + "incewptionebranchpool1").bulid(branchpool)

        output = [branch1x1_1, branch3x3, branch3x3_5, branchpool_1]
        # concancate
        IncB = concatenate(output, axis=3, name=self.name)
        return IncB

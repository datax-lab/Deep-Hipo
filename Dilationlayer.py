from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers import  BatchNormalization,Activation,Dropout,SpatialDropout2D
from keras.regularizers import l2

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# #The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
class IndiviualCNN():

    def __init__(self, filters, num_row, num_col, name, strides=(1, 1)):
        self.strides = strides
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.name = name
        self.conv_name = self.name + '_conv'
        self.bn_name = self.name + '_bn'
        self.Padding = "same"
        self.dilation_rate = (1, 1)
        self.eps = 0.001
        self.conv_name = self.name + '_conv'
        self.Kernel_regularizer = l2(0.02)
    def bulid(self, inputlayer):
        Dilatedlayer = Conv2D(self.filters, (self.num_row, self.num_col), dilation_rate=self.dilation_rate,
                              strides=self.strides, padding=self.Padding, use_bias=False, name=self.conv_name)(
            inputlayer)
        BatchNor = BatchNormalization(axis=1, scale=True, epsilon=self.eps, name=self.bn_name)(Dilatedlayer)

        BatchNor = SpatialDropout2D(0.3)(BatchNor)
        Act = Activation('relu', name=self.name)(BatchNor)
        return Act
class Indiviualdilation():

    def __init__(self, filters,num_row, num_col,name):
        self.filters = filters
        self.num_row = num_row
        self.num_col = num_col
        self.name = name
        self.conv_name = self.name+'_conv'
        self.bn_name = self.name+'_bn'
        self.Padding = "valid"
        self.dilation_rate = (2,2)
        self.eps = 0.001
        self.conv_name = self.name+'_conv'
        self.Kernel_regularizer = l2(0.02)
#         self.Dil = Conv2D(filters, (num_row,num_col),dilation_rate = self.dilation_rate,strides=(1,1),padding=self.Padding,use_bias=False,name = self.conv_name)
# #         self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#         self.bn = BatchNormalization(axis = 1, scale= True, epsilon = self.eps, name = self.bn_name)
#         self.Act = Activation('relu', name= name)
        
        
    def bulid(self,inputlayer):
    
        Dilatedlayer = Conv2D(self.filters, (self.num_row,self.num_col),dilation_rate = self.dilation_rate,strides=(1,1),padding=self.Padding,use_bias=False,name = self.conv_name, kernel_regularizer = self.Kernel_regularizer)(inputlayer)
        BatchNor = BatchNormalization(axis = 1, scale= False,  name = self.bn_name)(Dilatedlayer)
        BatchNor = SpatialDropout2D(0.4)(BatchNor)
        Act = Activation('relu', name= self.name)(BatchNor)
        return Act
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return F.relu(x, inplace=True)
class Dilationlayer():
    
    def __init__(self, name):
        
        self.name = name
#         self.dil1 = Indiviualdilation(32,3,3,name = self.name+"dil1")
#         self.dil2 = Indiviualdilation(32,3,3,name = name+"dil2")
#         self.dil3 = Indiviualdilation(64,3,3,name = name+"dil3")
# #         self.maxpooling = layers.MaxPooling2D((3, 3), strides=(2, 2))
#         self.dil4 = Indiviualdilation(64,1,1,name = name+"dil4")
#         self.dil5 = Indiviualdilation(80,3,3,name = name+"dil5")
        
    def bulid(self,inputmodel):
        Dil1 = Indiviualdilation(32,3,3,name = self.name+"dil1").bulid(inputmodel)
        Dil2 = Indiviualdilation(32,3,3,name = self.name+"dil2").bulid(Dil1)
        Dil3 = Indiviualdilation(64,3,3,name = self.name+"dil3").bulid(Dil2)
        Max1 = MaxPooling2D((3, 3),strides=(2, 2),padding='same')(Dil3)
        Dil4 = Indiviualdilation(64,1,1,name = self.name+"dil4").bulid(Max1)
        Dil5 = Indiviualdilation(80,3,3,name = self.name+"dil5").bulid(Dil4)
        Max2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Dil5)
        return Max2


class CNNlayer():

    def __init__(self, name):
        self.name = name

    #         self.dil1 = Indiviualdilation(32,3,3,name = self.name+"dil1")
    #         self.dil2 = Indiviualdilation(32,3,3,name = name+"dil2")
    #         self.dil3 = Indiviualdilation(64,3,3,name = name+"dil3")
    # #         self.maxpooling = layers.MaxPooling2D((3, 3), strides=(2, 2))
    #         self.dil4 = Indiviualdilation(64,1,1,name = name+"dil4")
    #         self.dil5 = Indiviualdilation(80,3,3,name = name+"dil5")

    def bulid(self, inputmodel):
        Dil1 = IndiviualCNN(32, 3, 3, name=self.name + "dil1").bulid(inputmodel)
        Dil2 = IndiviualCNN(32, 3, 3, name=self.name + "dil2").bulid(Dil1)
        Dil3 = IndiviualCNN(64, 3, 3, name=self.name + "dil3").bulid(Dil2)
        Max1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Dil3)
        Dil4 = IndiviualCNN(64, 1, 1, name=self.name + "dil4").bulid(Max1)
        Dil5 = IndiviualCNN(80, 3, 3, name=self.name + "dil5").bulid(Dil4)
        Max2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Dil5)
        return Max2
# global backend, layers, models, keras_utils
# backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
# def conv2d_bn(x,
#               filters,
#               num_row,
#               num_col,
#               padding='same',
#               strides=(1, 1),
#               name=None,
#               dilation_rate=(1, 1)):

#     if name is not None:
#         bn_name = name + '_bn'
#         conv_name = name + '_conv'
#     else:
#         bn_name = None
#         conv_name = None
#     if backend.image_data_format() == 'channels_first':
#         bn_axis = 1
#     else:
#         bn_axis = 3
#     x = layers.Conv2D(
#         filters, (num_row, num_col),
#         strides=strides,
#         padding=padding,
#         use_bias=False,
#         name=conv_name,dilation_rate=dilation_rate)(x)
#     x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
#     x = layers.Activation('relu', name=name)(x)
#     return x


# def CATNeT(include_top=False,
#                 weights=None,
#                 input_tensor=None,
#                 input_shape=None,
#                 pooling='max',
#                 classes=1000,
#                 **kwargs):
   


#     if not (weights in {'imagenet', None} or os.path.exists(weights)):
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization), `imagenet` '
#                          '(pre-training on ImageNet), '
#                          'or the path to the weights file to be loaded.')

#     if weights == 'imagenet' and include_top and classes != 1000:
#         raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
#                          ' as true, `classes` should be 1000')

#     # Determine proper input shape
#     input_shape = _obtain_input_shape(
#         input_shape,
#         default_size=299,
#         min_size=75,
#         data_format=backend.image_data_format(),
#         require_flatten=include_top,
#         weights=weights)

#     if input_tensor is None:
#         img_input = layers.Input(shape=input_shape)
#     else:
#         if not backend.is_keras_tensor(input_tensor):
#             img_input = layers.Input(tensor=input_tensor, shape=input_shape)
#         else:
#             img_input = input_tensor

#     if backend.image_data_format() == 'channels_first':
#         channel_axis = 1
#     else:
#         channel_axis = 3

#     x = conv2d_bn(img_input, 32, 3, 3, padding='valid', dilation_rate=(3, 3))
#     x = conv2d_bn(x, 32, 3, 3, padding='valid', dilation_rate=(3, 3))
#     x = conv2d_bn(x, 64, 3, 3,    x = conv2d_bn(x, 32, 3, 3, padding='valid',dilation_rate=(3, 3))
# )
#     x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

#     x = conv2d_bn(x, 80, 1, 1, padding='valid')
#     x = conv2d_bn(x, 192, 3, 3, padding='valid')
#     x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

#     # mixed 0: 35 x 35 x 256
#     branch1x1 = conv2d_bn(x, 64, 1, 1)

#     branch5x5 = conv2d_bn(x, 48, 1, 1)
#     branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

#     branch_pool = layers.AveragePooling2D((3, 3),
#                                           strides=(1, 1),
#                                           padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
#     x = layers.concatenate(
#         [branch1x1, branch5x5, branch3x3dbl, branch_pool],
#         axis=channel_axis,
#         name='mixed0')

#     # mixed 1: 35 x 35 x 288
#     branch1x1 = conv2d_bn(x, 64, 1, 1)

#     branch5x5 = conv2d_bn(x, 48, 1, 1)
#     branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

#     branch_pool = layers.AveragePooling2D((3, 3),
#                                           strides=(1, 1),
#                                           padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
#     x = layers.concatenate(
#         [branch1x1, branch5x5, branch3x3dbl, branch_pool],
#         axis=channel_axis,
#         name='mixed1')

#     # mixed 2: 35 x 35 x 288
#     branch1x1 = conv2d_bn(x, 64, 1, 1)

#     branch5x5 = conv2d_bn(x, 48, 1, 1)
#     branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

#     branch_pool = layers.AveragePooling2D((3, 3),
#                                           strides=(1, 1),
#                                           padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
#     x = layers.concatenate(
#         [branch1x1, branch5x5, branch3x3dbl, branch_pool],
#         axis=channel_axis,
#         name='mixed2')

#     # mixed 3: 17 x 17 x 768
#     branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

#     branch3x3dbl = conv2d_bn(x, 64, 1, 1)
#     branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
#     branch3x3dbl = conv2d_bn(
#         branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

#     branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
#     x = layers.concatenate(
#         [branch3x3, branch3x3dbl, branch_pool],
#         axis=channel_axis,
#         name='mixed3')

#     # mixed 4: 17 x 17 x 768
#     branch1x1 = conv2d_bn(x, 192, 1, 1)

#     branch7x7 = conv2d_bn(x, 128, 1, 1)
#     branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
#     branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

#     branch7x7dbl = conv2d_bn(x, 128, 1, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

#     branch_pool = layers.AveragePooling2D((3, 3),
#                                           strides=(1, 1),
#                                           padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#     x = layers.concatenate(
#         [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#         axis=channel_axis,
#         name='mixed4')

#     # mixed 5, 6: 17 x 17 x 768
#     for i in range(2):
#         branch1x1 = conv2d_bn(x, 192, 1, 1)

#         branch7x7 = conv2d_bn(x, 160, 1, 1)
#         branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
#         branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

#         branch7x7dbl = conv2d_bn(x, 160, 1, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
#         branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

#         branch_pool = layers.AveragePooling2D(
#             (3, 3), strides=(1, 1), padding='same')(x)
#         branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#         x = layers.concatenate(
#             [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#             axis=channel_axis,
#             name='mixed' + str(5 + i))

#     # mixed 7: 17 x 17 x 768
#     branch1x1 = conv2d_bn(x, 192, 1, 1)

#     branch7x7 = conv2d_bn(x, 192, 1, 1)
#     branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
#     branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

#     branch7x7dbl = conv2d_bn(x, 192, 1, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
#     branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

#     branch_pool = layers.AveragePooling2D((3, 3),
#                                           strides=(1, 1),
#                                           padding='same')(x)
#     branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#     x = layers.concatenate(
#         [branch1x1, branch7x7, branch7x7dbl, branch_pool],
#         axis=channel_axis,
#         name='mixed7')

#     # mixed 8: 8 x 8 x 1280
#     branch3x3 = conv2d_bn(x, 192, 1, 1)
#     branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
#                           strides=(2, 2), padding='valid')

#     branch7x7x3 = conv2d_bn(x, 192, 1, 1)
#     branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
#     branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
#     branch7x7x3 = conv2d_bn(
#         branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

#     branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
#     x = layers.concatenate(
#         [branch3x3, branch7x7x3, branch_pool],
#         axis=channel_axis,
#         name='mixed8')

#     # mixed 9: 8 x 8 x 2048
#     for i in range(2):
#         branch1x1 = conv2d_bn(x, 320, 1, 1)

#         branch3x3 = conv2d_bn(x, 384, 1, 1)
#         branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
#         branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
#         branch3x3 = layers.concatenate(
#             [branch3x3_1, branch3x3_2],
#             axis=channel_axis,
#             name='mixed9_' + str(i))

#         branch3x3dbl = conv2d_bn(x, 448, 1, 1)
#         branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
#         branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
#         branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
#         branch3x3dbl = layers.concatenate(
#             [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

#         branch_pool = layers.AveragePooling2D(
#             (3, 3), strides=(1, 1), padding='same')(x)
#         branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
#         x = layers.concatenate(
#             [branch1x1, branch3x3, branch3x3dbl, branch_pool],
#             axis=channel_axis,
#             name='mixed' + str(9 + i))
#     if include_top:
#         # Classification block
#         x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
#         x = layers.Dense(classes, activation='softmax', name='predictions')(x)
#     else:
#         if pooling == 'avg':
#             x = layers.GlobalAveragePooling2D()(x)
#         elif pooling == 'max':
#             x = layers.GlobalMaxPooling2D()(x)

#     # Ensure that the model takes into account
#     # any potential predecessors of `input_tensor`.
#     if input_tensor is not None:
#         inputs = keras_utils.get_source_inputs(input_tensor)
#     else:
#         inputs = img_input
#     # Create model.
#     model = models.Model(inputs, x, name='CATNeT')

#     return model


# def preprocess_input(x, **kwargs):
#     """Preprocesses a numpy array encoding a batch of images.

#     # Arguments
#         x: a 4D numpy array consists of RGB values within [0, 255].

#     # Returns
#         Preprocessed array.
#     """
#     return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)

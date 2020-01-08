import pandas as pd
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from keras import optimizers
import os
import random
from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
#random.seed(42)
import os
from os import walk
# from Datageneator import data_generator
from keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.callbacks import EarlyStopping, ModelCheckpoint
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from Models import inceptionv3_model,DoTNetmodel,Vgg19
from DataGenetors import ImgDataParameters,DataGenerator
from CAT_Net2 import CATNet2
inputdir = "/home/skosaraju/Expirementsdata"
# ##Data_Genetor
'''
datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=True)
#
train_generator = datagen.flow_from_directory(
    directory=r"/home/skosaraju/Expirementsdata/Train_ck_pd",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=32,
    class_mode='binary',
    shuffle=True,
    seed=42
)

valid_generator = datagen.flow_from_directory(
    directory=r"/home/skosaraju/Expirementsdata/Valid_ck_pd",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=1,
    class_mode='binary',
    shuffle=True,
    seed=42
)
#
test_generator = datagen.flow_from_directory(
    directory=r"/home/skosaraju/Expirementsdata/Test_ck_pd",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=1,
    class_mode='binary',
    shuffle=False,
    seed=42
)
'''
input_imgen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2,
                                   rotation_range=5.,
                                   horizontal_flip = True)

test_imgen = ImageDataGenerator(rescale = 1./255)



def generate_generator_multiple(generator,dir1, dir2):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (299,299),
                                          class_mode = 'binary',
                                          color_mode="rgb",
                                          batch_size = 1,
                                          shuffle=False, 
                                          seed=42)
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (299,299),
                                          class_mode = 'binary',
                                          color_mode="rgb",
                                          batch_size = 1,
                                          shuffle=False, 
                                          seed=42)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label
            
            
traingenerator=generate_generator_multiple(generator=input_imgen,
                                           dir1= r"/home/skosaraju/Expirementsdata/CATNET/TrainMD",
                                           dir2= r"/home/skosaraju/Expirementsdata/CATNET/TrainMD5x",
                                          )
     

validationgenerator = generate_generator_multiple(input_imgen,
                                          dir1= r"/home/skosaraju/Expirementsdata/CATNET/ValidMD",
                                          dir2= r"/home/skosaraju/Expirementsdata/CATNET/ValidMD5x",
                                          )
     
testgenerator=generate_generator_multiple(test_imgen,
                                            dir1= r"/home/skosaraju/Expirementsdata/CATNET/TestMD",
                                            dir2= r"/home/skosaraju/Expirementsdata/CATNET/TestMD5x",
                                                                                    )
          


batch_size = 1
# print([train_generator.__getitem__(0)[0]])
Lr = [0.001,0.0015,0.0001,0.00015]
# Lr = [0.001]
Beta_1 = [0.8,0.85,0.9]
# Beta_1 = [0.85,0.9]
Paramaters_list = []
for lr in  Lr:
    for beta1 in Beta_1:
        Adam = optimizers.adam(lr=lr , beta_1 = beta1)
        InputA = Input(shape=(299, 299,3))
        InputB = Input(shape=(299, 299,3))
        model = CATNet2.bulid(InputA, InputB)

#         model =  CATNet2()
        # print(model.summary())
        # model = CATNet2()
        model.compile(optimizer = Adam, loss = 'mean_squared_error',metrics=['accuracy'])
        #fmodel.fit(X1_train, y_train, validation_split=0.15, batch_size=256, epochs=100)

        #STEP_SIZE_TRAIN=traingenerator.n//train_generator.batch_size
        #STEP_SIZE_VALID=validationgenerator.n//valid_generator.batch_size
        #STEP_SIZE_TEST = testgenerator.n//test_generator.batch_size
        callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        #print(len(traingenerator.next()))
        epochs = 32
        model.fit_generator(traingenerator,
                        steps_per_epoch=1000,
                        epochs = epochs,
                        validation_data = validationgenerator,

                        validation_steps = 100,
                        use_multiprocessing=True,
                        shuffle=False)


        # model.save('ckinception.h5')
        # model.evaluate_generator(generator=validationgenerator,steps =STEP_SIZE_VALID )
        y_valid = model.predict_generator(validationgenerator,steps = STEP_SIZE_VALID )
        y_test = [validationgenerator.__getitem__(i)[1] for i in range(validationgenerator.__len__())]

        accuracyscore = accuracy_score(y_test,y_valid>0.5)
        print("validation_accuracy for lr=%s beta_1=%s : %s"%(lr,beta1,accuracyscore))
        Paramaters_list.append((accuracyscore,lr,beta1))
Paramaters_list.sort(reverse = True)
print(Paramaters_list)
print("___________________________")
print("Optimal_parameters lr: %s beta_1 : %s"% (Paramaters_list[0][1],Paramaters_list[0][2]))
Adam = optimizers.adam(lr= Paramaters_list[0][1], beta_1 = Paramaters_list[0][2])
model.compile(optimizer = Adam, loss = 'mean_squared_error',metrics=['accuracy'])
#STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
#STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
#STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
model.fit_generator(generator=traingenerator,
                    steps_per_epoch=1247,
                                   epochs=50
                                        )
model.save('CATNET2MD.h5')
model.evaluate_generator(generator=testgenerator, steps=STEP_SIZE_TEST)
y_pred = model.predict_generator(testgenerator, steps=STEP_SIZE_TEST, verbose=1)
#y_test = [testgenerator.__getitem__(i)[1] for i in range(testgenerator.__len__())]

#accuracyscore = accuracy_score(y_test, y_pred > 0.5)
print("final_accuracy: %s" % accuracyscore)






labelpredict = []
labeltest= []


    # print(test_generator)
labelpredict.extend(y_pred)

labeltest.extend(y_test)
labelpredictdf = pd.DataFrame(labelpredict)
labelpredictdf.to_csv("labelpredictcat.csv")
#labeltestdf = pd.DataFrame(labeltest)
#labeltestdf.to_csv("labeltestcat.csv")
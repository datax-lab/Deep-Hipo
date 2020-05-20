# Deep learning model for HIstoPatholOgy (Deep-Hipo)
Deep-Hipo is designed to accurately analyze histopathological images by learning multi-scale morphological patterns from various magnification levels of patches in a WSI simultaneously

![Alt text](Deep-Hipo/figures/Deep_Hipo_overview.png?raw=true "Framework")


## Required packages:
> - PyHistopathology
> - Open CV
> - Sklearn
> - Keras
> - Tensorflow

## CODE FOLLOW:
Codes for Training are in Deep-Hipo folder
> - **_Deep-HipoTrain.py_** is the interface for train our model. It has data generators which automatically loads multiple inputs simantionously. 
> - **Deep-Hipo_Model.py** Constructs Deep-Hipo architecture.
> - **Dilationlayer.py** Dilation Layers (intial layers) for Deep-Hipo architecture
> - **InceptionA.py** Module A of our architecture.
> - **InceptionB.py** Module B of our architecture.
> - **InceptionC.py** Module C of our architecture.
> - **InceptionD.py** Module D of our architecture.
> - **InceptionE.py** Module E of our architecture.

![Alt text](Deep-Hipo/figures/fig3.PNG?raw=true "Framework")

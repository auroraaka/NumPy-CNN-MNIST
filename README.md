# Pure-NumPy-CNN-for-MNIST-Dataset

A pure NumPy implementation of a convolutional neural network for classification of MNIST handwritten numbers dataset. This project was completed as a learning exercise to gain an intuition for mathematical fundamentals of machine learning. All hyperparameters and optimizers are provided in the source code and Google Colaboratory. The model trained on five epochs with the below architecture achieved an accuracy of 97.01% on the test dataset. 

```
Model: CNN_MNIST
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 
 conv2d (Conv2D)           (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling  (None, 13, 13, 32)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)         (None, 5408)              0         

 dropout (Dropout)         (None, 5408)              0
                                                                 
 dense (Dense)             (None, 100)               540900    
                                                                 
 dense (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 542,230
Trainable params: 542,230
Non-trainable params: 0
_________________________________________________________________
```

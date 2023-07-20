# NumPy CNN MNIST

A pure NumPy implementation of a convolutional neural network for classification of the MNIST handwritten numbers dataset. This project was completed as a learning exercise to gain an intuition for mathematical fundamentals of machine learning. All the hyperparameters and optimization algorithms used in this project can be found in the provided source code and the associated Google Colaboratory notebook. The described architecture of the model, which was trained over five epochs, resulted in a classification accuracy of 97.01% on the test dataset.

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

# VectorClassifier
Simple classifier for vector data implemented on TensorFlow

## Overview
This project will consist of three programs.

* _VectorClassifier.py_  
    Main program of this project. Imeplementation class to classify number vector data.
* _Trainer.py_  
    Sample program to run training. It requires training vector csv data, which will be described below.
* _Predictor.py_  
    _This program is currently under implementation._ Sample program to predict class by using trained data.

## VectorClassifier.py
Python class which has training and prediction functions of vector classification.  
This class is implemented based on framework TensorFlow.

Currently, classification is implemented on Neural Network and softmax.

### Input
### Training data
This classifier requires CSV file listing _label_ and _vector_ in each row as training data.
In each row, the first column value is treated as label, "1" or "0".
The other columns are treated as vector.


In the example below, vector [1.2, 0.5, -0.2] belong to class "1", and vector [0.0, 10.5, 0.7] belong to class "0".
``


1, 1.2, 0.5, -0.2


0, 0.0, 10.5, 0.7


``

This repository include sample csv in directory _data_.

### Output
While training, this classifier output the status and result of learning as below.

#### Standart output
While training, learning progress is written to standard output:


_Step: 2500, Loss: 0.011809 @ 2017/11/06 01:31:40_

This consists of 3parts:  
* Step number in learning process
* Loss value, which is sequare sum of value differences between predicted vector and training vector
* Timestamp when the progress is output  

Default implementation of running parts write progress at the first step and every 500 steps after the first step.

#### Tensor Board
_VectorClassifier.py_ outputs files for _Tensor Board_ in _board_ directory.  
You can see _Tensor Board_ by _Tensorflow_'s usual way to invoke it.

_TODO:to be written for more details._

#### Session file to resume learning and prediction
_Trainer.py_ saves files of session files, which includes trained parameters at the saving step.  
These files are saved in _saved_session_ directory.

_Trainer.py_ can load them to resume training from saved step.  
_Predictor.py_ will be able to load them to predict classes by using trained parameters at saved step.  

## Trainer.py
Sample implementation of training by using VectorClassifier.py.
This implementation use data from  _data/data01.csv_.  

_TODO:to be written about more details._

## Predictor.py
Sample implementation of prediction by using VectorClassifier.py.
This implementation use data from  _predictList.csv_.  

_TODO:to be implemented._


_TODO:to be written about more details._

## Requirements
* Python3, Tensorflow 1.1.0, and the libraries it requires.

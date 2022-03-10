
# Utilize PEP-8 package import coding standard

# Standard Library Packages First
from datetime import datetime
import logging
import os
import sys

# 3rd Party Packages Second
import configargparse
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import seaborn
import textwrap

# Local Packages Third

def create_parser():
    parser = configargparse.ArgParser(default_config_files=['./paxaGUI_config.txt'],
       
        description='Generate a Deep Feed Forward Neural Network script using SciKit-Learn, Keras, and Tensorflow (backend) with parameters defined by command line arguments.',
        epilog='Example: python Train_Neural_Network.py -inp file.csv -cols 6 -L 10 -N 512 -E 10'
    parser.add_argument('-c', '--config', required = False, is_config_file = True, help = 'Defines the configuration file path')
    # TODO: Include options for user to use CPU or GPU Tensorflow backend
    # parser.add_argument('-cuda', '--cuda-settings', type=bool, required=False, default = True,
    #                     help='CUDA settings used to specify if CPU or GPU should be used. Default is GPU.')
    parser.add_argument('-inp', '--input-file', type=str, required=True,
                        help='Input file in CSV format to be used for training. Data should inlude n-inputs (data features) and 1-output. The output is required to be the last column in the CSV file.')
    parser.add_argument('-cols', '--training-columns', type=int, required=True,
                        help='Number of input columns (data features) to use for training.')
    parser.add_argument('-ts', '--test-size', type=float, required=False, default=0.25,
                        help=textwrap.dedent('''\
                        Percentage of data to split for testing purposes. Should be in the range between 0.1 to 0.3 (10-30 percent)
                        Default is 0.25 (25 percent)'''))

    # SciKit-Learn Scaling Functions.
    # Standard Scaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    parser.add_argument('-s', '--scaling-type', type=int, required=False, default=3,
                        help=textwrap.dedent('''\
                        Type of data scaling method to be used.
                        Default is MinMaxScaler from 0 to 1
                        1: Standard Scaler - Centers and scales data to have a mean of zero and unit variance
                        2: MinMaxScaler from -1 to 1. Transforms feature by scaling to the given range
                        3: MinMaxScaler from 0 to 1. Transforms feature by scaling to the given range'''))

    parser.add_argument('-bs', '--batch-size', type=int, required=False, default=32,
                        help=textwrap.dedent('''\
                        Training batch size used during training.
                        Default is batch size of 32'''))
    parser.add_argument('-L', '--layers', type=int, required=True,
                        help='Number of hidden layers for the neural network')
    parser.add_argument('-N', '--nodes', type=int, required=True,
                        help='Number of nodes per layer for the neural network')
    parser.add_argument('-E', '--epochs', type=int, required=True,
                        help='Number of epochs to be used for the neural network')

    # Keras activation function information https://keras.io/activations/
    parser.add_argument('-af', '--activation-function', type=int, required=False, default=6,
                        help=textwrap.dedent('''\
                        Type of activation function to use for the neural network. Default is RELU
                        1: ELU - Exponential Linear Unit
                        2: Softmax
                        3: SELU - Scaled Exponential Linear Unit
                        4: Softplus
                        5: Softsign
                        6: RELU - Rectified Linear Unit (Default)
                        7: TANH - Hyperbolic tangent activation function
                        8: Sigmoid
                        9: Hard Sigmoid
                        10: Exponential - Exponential (base e) activation function
                        11: Linear - Linear (i.e. identity) activation function'''))

    # Keras optimization function information https://keras.io/optimizers/
    parser.add_argument('-opt', '--optimization-function', type=int, required=False, default=5,
                        help=textwrap.dedent('''\
                        Type of optimization function to use for the neural network. Default is Adam
                        1: SGD - Stochastic Gradient Descent
                        2: RMSprop - Root Mean Squared prop
                        3: Adagrad
                        4: Adadelta
                        5: Adam (Default)
                        6: Adamax
                        7: Nadam - Nesterov Adam optimizer'''))

    # Keras learning rate for chosen optimization function
    parser.add_argument('-lr', '--learning-rate', type=float, required=False,
                        help=textwrap.dedent('''\
                        Learning rate to use for the optimization algorithm. For better results,
                        a learning rate is set to the default and then gradually reduced by an order of magnitude
                        '''))

    # Keras loss function information https://keras.io/losses/
    parser.add_argument('-loss', '--loss-function', type=int, required=False, default=1,
                        help=textwrap.dedent('''\
                        Type of loss function to use for the neural network. Default is MSE
                        1: MSE - Mean Squared Error (Default)
                        2: MAE - Mean Absolute Error
                        3: MAPE - Mean Absolute Percentage Error
                        4: MSLE - Mean Squared Logarithmic Error
                        5: Squared Hinge
                        6: Hinge
                        7: Categorical Hinge
                        8: LogCosh - Logarithm of the hyperbolic cosine of prediction error
                        9: Huber Loss
                        10: Categorical Cross Entropy
                        11: Sparse Categorical Cross Entropy
                        12: Binary Cross Entropy
                        13: Kullback Leibler Divergence
                        14: Poisson
                        15: Cosine Proximity
                        16: Is Categorical Cross Entropy (Requires n-dimensional vectors that is all zeros except for a 1 at index'''))

    return parser
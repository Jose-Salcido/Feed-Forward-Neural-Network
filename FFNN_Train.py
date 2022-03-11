
# Utilize PEP-8 package import coding standard

# Standard Library Packages First
import ast
from datetime import datetime
import logging
import os
import sys
from pathlib import Path

# 3rd Party Packages Second
import configargparse
import matplotlib.pyplot as plt
import joblib
#from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import pandas as pd
from scipy import stats
import seaborn
from sklearn.model_selection import train_test_split
from keras import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import textwrap
      
# Local Packages Third

'''
Ignore Tensorflow warning messages from out depricated libraries. Only allow error warnings
'''
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  

'''
Set CUDA settings
'''
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Used to force CPU usage
# Dynamically grow the memory used on the GPU
config = tf.compat.v1.ConfigProto()
tf.compat.v1.GPUOptions(allow_growth = True)

# Logs device placement (on which device the operation ran)
# config.log_device_placement = True # Verbose display of device placement. Use for debugging purposes
# (nothing gets printed in Jupyter, only if you run it standalone)
current_session = tf.compat.v1.Session(config=config)
# Set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(current_session)


def create_parser():
    parser = configargparse.ArgParser(default_config_files=['./ffnn_config.txt'],   
        description='Train a Feed Forward Neural Network utilizing SciKit-Learn, Keras, and Tensorflow (used for backend) with parameters defined by command line arguments.',
        epilog='Example: python FFNN_Train.py -inp <input_file.csv> -cols <int> -L <int> -N <int> -E <int>')
    parser.add_argument('-c', '--config', required = False, is_config_file = True, help = 'Defines the configuration file path')
    # TODO: Include options for user to use CPU or GPU Tensorflow backend
    # parser.add_argument('-cuda', '--cuda-settings', type=bool, required=False, default = True,
    #                     help='CUDA settings used to specify if CPU or GPU should be used. Default is GPU.')
    parser.add_argument('-inp', '--input_file', type=str, required=False,
                        help='Input file in CSV format to be used for training. Data should inlude n-inputs (data features) and 1-output. The output is required to be the last column in the CSV file.')
    parser.add_argument('-cols', '--training_columns', type=int, required=False,
                        help='Number of input columns (data features) to use for training.')
    parser.add_argument('-ts', '--test_size', type=float, required=False, default=0.25,
                        help=textwrap.dedent('''\
                        Percentage of data to split for testing purposes. Should be in the range between 0.1 to 0.3 (10-30 percent)
                        Default is 0.25 (25 percent)'''))

    # SciKit-Learn Scaling Functions.
    # Standard Scaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # MinMaxScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    parser.add_argument('-s', '--scaling_type', type=int, required=False, default=3,
                        help=textwrap.dedent('''\
                        Type of data scaling method to be used.
                        Default is MinMaxScaler from 0 to 1
                        1: Standard Scaler - Centers and scales data to have a mean of zero and unit variance
                        2: MinMaxScaler from -1 to 1. Transforms feature by scaling to the given range
                        3: MinMaxScaler from 0 to 1. Transforms feature by scaling to the given range'''))
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=32,
                        help=textwrap.dedent('''\
                        Training batch size used during training.
                        Default is batch size of 32'''))
    parser.add_argument('-L', '--layers', type=int, required=False,
                        help='Number of hidden layers for the neural network')
    parser.add_argument('-N', '--nodes', type=int, required=False,
                        help='Number of nodes per layer for the neural network')
    parser.add_argument('-E', '--epochs', type=int, required=False,
                        help='Number of epochs to be used for the neural network')

    # Keras activation function information https://keras.io/activations/
    parser.add_argument('-af', '--activation_function', type=int, required=False, default=6,
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
    parser.add_argument('-opt', '--optimization_function', type=int, required=False, default=5,
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
    parser.add_argument('-lr', '--learning_rate', type=float, required=False,
                        help=textwrap.dedent('''\
                        Learning rate to use for the optimization algorithm. For better results,
                        a learning rate is set to the default and then gradually reduced by an order of magnitude
                        '''))

    # Keras loss function information https://keras.io/losses/
    parser.add_argument('-loss', '--loss_function', type=int, required=False, default=1,
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


def GenDirStructure():
    current_dir = os.getcwd()
    dir_path = current_dir + r"\NeuralNetworkResults_" + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    if not os.path.exists(dir_path):
        print("Generating new directory to store Neural Network training results")
        os.makedirs(dir_path)

    #os.chdir(dir_path)

    return current_dir

class PlottingFunctions:


    def Histogram(self, output_col_name, y_output_data):
        bins = 100
        plt.figure(figsize=(15, 10))
        plt.title('Histogram Plot: Output Data Distribution ' + '(bins=' + str(bins) + ')', fontsize=20, weight='bold')
        plt.ylabel('Counts', fontsize=14)
        plt.xlabel('Output Data Value', fontsize=20)
        plt.hist(y_output_data, bins=bins, label=output_col_name, alpha=0.8)
        plt.legend(fontsize=20)
        plt.savefig('Histogram_Output_Distribution.png')


    def CorrHeatMap(self, df):
        corr = df.corr()
        size = len(df.columns)
        fig, ax = plt.subplots(figsize=(4*size, 4*size))  # Dynamically change size of plot
        ax.set_title('Data Correlation Heat Map', fontsize=20, weight='bold')
        colormap = seaborn.diverging_palette(220, 10, as_cmap=True)
        seaborn.heatmap(corr, cmap=colormap, annot=True, fmt=".4f")
        ax.set_yticklabels(labels=corr.columns, va='center')
        ax.set_xticklabels(labels=corr.columns, ha='center')
        plt.savefig('Data_Correlation_HeatMap.png')


    def LossPlot(self, history):
        plt.figure(figsize=(15, 10))
        plt.title('Loss', fontsize=20, weight='bold')
        plt.ylabel('Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        loss = np.min(history.history['loss'])
        logger.info("Training Data Loss (Obj Function): " + str(loss))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend(fontsize=20)
        plt.savefig("Loss.png")


    def MSEPlot(self, history):
        plt.figure(figsize=(15, 10))
        plt.title('Mean Squared Error', fontsize=20, weight='bold')
        plt.ylabel('MSE', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        plt.plot(history.history['mean_squared_error'], label='train')
        plt.plot(history.history['val_mean_squared_error'], label='test')
        plt.legend(fontsize=20)
        plt.savefig("Mean_Squared_Error.png")


    def MAEPlot(self, history):
        plt.figure(figsize=(15, 10))
        plt.title('Mean Absolute Error', fontsize=20, weight='bold')
        plt.ylabel('MAE', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        plt.plot(history.history['mean_absolute_error'], label='train')
        plt.plot(history.history['val_mean_absolute_error'], label='test')
        plt.legend(fontsize=20)
        plt.savefig("Mean_Absolute_Error.png")


    def LogLogPlot(self, y_test, Unscaled_y_Predictions, output_col_name):
        x = y_test  # Set x to be the truth data
        y = Unscaled_y_Predictions  # Set y to be the prediction data
        plt.figure(figsize=(15, 10))
        plt.title('Log-Log Plot: Predicted vs. Truth', fontsize=20, weight='bold')
        plt.ylabel('Predicted ' + str(output_col_name), fontsize=20)
        plt.xlabel('Truth ' + str(output_col_name), fontsize=20)
        plt.loglog(x, y, 'o', markersize=4, alpha=0.7)
        plt.savefig("LogLogPlot_Truth_vs_Predicted.png")


    def ScatterPlot(self, y_test, Unscaled_y_Predictions, output_col_name):
        x = y_test  # Set x to be the truth data
        y = Unscaled_y_Predictions  # Set y to be the prediction data
        # Calculate the regression line with scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[:, 0], y[:, 0])
        yhat = intercept + slope * x
        plt.figure(figsize=(15, 10))
        plt.title('Scatter Plot: Predicted vs. Truth', fontsize=20, weight='bold')
        plt.ylabel('Predicted ' + str(output_col_name), fontsize=20)
        plt.xlabel('Truth ' + str(output_col_name), fontsize=20)
        plt.scatter(x, y, s=5, alpha=0.7)
        plt.plot(x, yhat, color='g', label='regression line')  # Over-plot the regression line in green
        plt.plot((np.min(y), np.max(x)), (np.min(y), np.max(x)), 'r-',
                 label='truth line')  # Over-plot the truth line in red
        r2 = r_value ** 2
        Pval = p_value
        logger.info("Scatter Plot with Regression Line:")
        logger.info('slope=%.3f, intercept=%.3f' % (slope, intercept))
        logger.info("Truth and Prediction Statistics:")
        logger.info('Predictions: mean=%.3f stdv=%.3f' % (np.mean(y), np.std(y)))
        logger.info('      Truth: mean=%.3f stdv=%.3f\n' % (np.mean(x), np.std(x)))
        logger.info(
            'Calculate R-squared which explains percent of model variance. Also known as coefficient of determination.')
        logger.info(
            'Value range 0-100%. An R-squared of 0% indicates model explains none of the variability of the response data around its mean.')
        logger.info('R-squared: ' + str(r2))
        # TODO: Research when it's acceptable to use P-value, T-Test
        # logger.info('Calculate p-value or probability value. This is the probability of obtaining test results at least as extreme as results actually observed.')
        # logger.info('Value range 0-100%. A p-value of 0% will allow us to reject the null-hypothesis. In this case, that our samples were generated by the normal distribution.')
        # logger.info('P-value: ' + str(Pval))
        plt.legend(fontsize=20)
        plt.savefig('ScatterPlot_Truth_vs_Predicted.png')


    def ResidualPlot(self, y_test, Unscaled_y_Predictions):
        x = y_test
        y = Unscaled_y_Predictions - y_test
        # y0 = 0
        # Calculate the regression line with scipy.stats.linregress
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[:, 0], y[:, 0])
        yhat = intercept + slope * x
        plt.figure(figsize=(15, 10))
        plt.title('Scatter Plot: Residual vs. Truth', fontsize=20, weight='bold')
        plt.ylabel('Residual (prediction - truth)', fontsize=20)
        # plt.xlabel('Truth ' + str(output_col_name), fontsize=20)
        plt.scatter(x, y, s=5, alpha=0.7)
        plt.plot(x, yhat, color='k', label='regression line')  # Over-plot the regression line in green
        plt.legend(fontsize=20)
        plt.savefig('ScatterPlot_Residual_vs_Truth.png')


class NeuralNetworkDict:
    

    def ScalingType(self, scaling_type):
        scalers_dict = {1: 'StandardScaler', 2: 'MinMaxScaler_11', 3: 'MinMaxScaler_01'}
        if scaling_type < 1 or scaling_type > 3:
            print('\nInput Error: Scaling type {scaling_type} must be a whole number between 1 and 3.  Use "python FFNN_Train.py -h" for help on usage.')
            sys.exit(1)
        elif scaling_type == 1:
            return scalers_dict.get(scaling_type)
        elif scaling_type == 2:
            return scalers_dict.get(scaling_type)
        elif scaling_type == 3:
            return scalers_dict.get(scaling_type)


    def ActivationFunction(self, activation_function):
        activation_functions_dict = {
                                        1: 'elu', 
                                        2: 'softmax',
                                        3: 'selu',
                                        4: 'softplus',
                                        5: 'softsign',
                                        6: 'relu',
                                        7: 'tanh',
                                        8: 'sigmoid',
                                        9: 'hard_sigmoid',
                                        10: 'exponential',
                                        11: 'linear'
                                    }
        if activation_function < 1 or activation_function > 11:
            print('\nInput Error: Activation Funtion {activation_function} must be a whole number between 1 and 11. Use "python FFNN_Train.py -h" for help on usage.')
            sys.exit(1)
        else:
            return activation_functions_dict.get(activation_function)


    def OptimizationFunction(self, optimization_function):
        optimization_functions_dict = {
                                        1: 'SGD', 
                                        2: 'RMSprop', 
                                        3: 'Adagrad', 
                                        4: 'Adadelta',
                                        5: 'Adam', 
                                        6: 'Adamax', 
                                        7: 'Nadam'
                                      }
        if optimization_function < 1 or optimization_function > 7:
            print('\nInput Error: Optimization Funtion {optimization_function} must be a whole number between 1 and 7. Use "python FFNN_Train.py -h" for help on usage.')
            sys.exit(1)
        else:
            return optimization_functions_dict.get(optimization_function)


    def LearningRate(self, learning_rate):
        learning_rate_dict = {
                                'SGD': 0.01, 
                                'RMSprop': 0.001, 
                                'Adagrad': 0.01, 
                                'Adadelta': 1.0,
                                'Adam': 0.001, 
                                'Adamax': 0.0020, 
                                'Nadam': 0.002
                            }
        return learning_rate_dict.get(learning_rate)


    def LossFunction(self, loss_function):
        loss_functions_dict = {
                                1: 'mean_squared_error', 
                                2: 'mean_absolute_error', 
                                3: 'mean_absolute_percentage_error',
                                4: 'mean_squared_logarithmic_error', 
                                5: 'squared_hinge', 
                                6: 'hinge', 
                                7: 'categorical_hinge', 
                                8: 'logcosh', 
                                9: 'huber_loss', 
                                10: 'categorical_crossentropy', 
                                11: 'sparse_categorical_crossentropy', 
                                12: 'binary_crossentropy', 
                                13: 'kullback_leibler_divergence',
                                14: 'poisson',
                                15: 'cosine_proximity',
                                16: 'is_categorical_crossentropy'
                               }
        if loss_function < 1 or loss_function > 16:
            print('\nInput Error: Loss Funtion type -loss, --loss-function, must be a whole number between 1 - 16. Use "python FFNN_Train.py -h" for help.')
            sys.exit(1)
        else:
            return loss_functions_dict.get(loss_function)


def remove_existing_files(data, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)  
        joblib.dump(data, output_file)
    else:
        joblib.dump(data, output_file)

def generate_scaler_output_data_files(scalerX, X_train, X_test, scalerY, y_train, y_test):
    Scaled_X_train = scalerX.fit_transform(X_train)
    Scaled_X_test = scalerX.fit_transform(X_test)
    Scaled_y_train = scalerY.fit_transform(y_train)
    Scaled_y_test = scalerY.fit_transform(y_test)

    return Scaled_X_train, Scaled_X_test, Scaled_y_train, Scaled_y_test


def train_neural_network(input_file,
                         training_columns, 
                         test_size, 
                         scaling_type, 
                         batch_size, 
                         layers,
                         nodes, 
                         epochs,
                         activation_function,
                         optimization_function,
                         learning_rate,
                         loss_function):

    logger.info('\n#=========================== TRAINING DATA, TEST SIZE, SCALING INFORMATION ==========================#\n')
    logger.info(f'Input File: {input_file}')

    if not os.path.exists(input_file):
        print(f'\nInput Error: The CSV file {input_file} was not found in the provided directory.')
        logger.warning(f'Input Error: The CSV file {input_file} was not found in the provided directory.\n')
        sys.exit(1)
    else:
        df = pd.read_csv(input_file, skipinitialspace=True)

    # Clean data if it contains nan or NA cells
    df = df.dropna()
    print(df)
    # Capture training features. all rows, column 0 through user defined training columns
    df_X = df.iloc[:, 0:training_columns]
    # Capture the output column which should be the last column in the CSV file
    df_y = df.iloc[:, [-1]]
    df_concat = pd.concat([df_X, df_y], axis=1)

    # Capture the output column name for plotting purposes.
    # The column name is captured as a numpy array.
    # We convert the array to a string by calling on the first 0th value
    output_col_name = df_y.columns.values
    output_col_name = output_col_name[0]
    logger.info(f'Output Column Name: {output_col_name}')

    # Split the data into training and testing data sets
    logger.info(f'Test Size: {test_size}')
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = test_size, shuffle = True)

    # Convert DataFrames to Numpy arrays. The Keras functions work with arrays.
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Convert output DataFrame to a Numpy array for plotting
    y_output_data = np.array(df_y)

    '''
    Define the scaling algorithm
    '''
    # Capture the user specified scaler if different from default of 1: StandardScaler
    user_defined_scaler = NeuralNetworkDict().ScalingType(scaling_type)
    logger.info(f'User Defined Scaling Algorithm: {user_defined_scaler}')

    scalerX_filename = "MinMaxScaler_scalerX.save"
    scalerY_filename = "MinMaxScaler_scalerY.save"

    Scaled_X_train = np.array
    Scaled_X_test = np.array
    Scaled_y_train = np.array
    Scaled_y_test = np.array

    if user_defined_scaler == 'StandardScaler':
        scaler = StandardScaler()
        print("Using StandardScaler")
        StandardScaler_filename = "StandardScaler.save"
        Scaled_X_train, Scaled_X_test, Scaled_y_train, Scaled_y_test = generate_scaler_output_data_files(scaler, X_train, X_test, scaler, y_train, y_test)
        remove_existing_files(scaler, StandardScaler_filename)

    elif user_defined_scaler == 'MinMaxScaler_11':
        scalerX, scalerY = preprocessing.MinMaxScaler(feature_range=(-1, 1)), preprocessing.MinMaxScaler(feature_range=(-1, 1))
        print("Using MinMaxScaler from -1 to 1")
        Scaled_X_train, Scaled_X_test, Scaled_y_train, Scaled_y_test = generate_scaler_output_data_files(scalerX, X_train, X_test, scalerY, y_train, y_test)
        remove_existing_files(scalerX, scalerX_filename)
        remove_existing_files(scalerY, scalerY_filename)

    elif user_defined_scaler == 'MinMaxScaler_01':
        scalerX, scalerY = preprocessing.MinMaxScaler(feature_range=(0, 1)), preprocessing.MinMaxScaler(feature_range=(0, 1))
        print("Using MinMaxScaler from 0 to 1")
        Scaled_X_train, Scaled_X_test, Scaled_y_train, Scaled_y_test = generate_scaler_output_data_files(scalerX, X_train, X_test, scalerY, y_train, y_test)
        remove_existing_files(scalerX, scalerX_filename)
        remove_existing_files(scalerY, scalerY_filename)

    # Capture the batch size, number of columns, layers, nodes, and epochs for the neural network
    n_cols = training_columns

    logger.info('\n#==================================== NEURAL NETWORK ARCHITECTURE ===================================#\n')

    logger.info(f'Number of Columns: {training_columns}')
    logger.info(f'Number of Layers: {layers}')
    logger.info(f'Number of Epochs: {epochs}')
    logger.info(f'Using Batch Size: {batch_size}')
    logger.info(f'Number of Nodes: {nodes}')
    print(f'Layers, Nodes, Epochs:  {layers}, {nodes}, {epochs}')

    '''
    DEFINE ACTIVATION FUNCTION
    '''
    # Construct the neural network layers
    # Capture the user defined activation function if different from default of RELU
    user_defined_activation_function = NeuralNetworkDict().ActivationFunction(activation_function)

    Input_1 = Input(shape=(n_cols,))
    layer = Dense(nodes, activation=user_defined_activation_function)(Input_1)
    for i in range(layers - 1):
        layer = Dense(nodes, activation=user_defined_activation_function)(layer)
    # Narrowing down number of nodes by half steps seems to improve training and predictions
    layer = Dense(int(nodes / 2), activation=user_defined_activation_function)(layer)
    layer = Dense(int(nodes / 4), activation=user_defined_activation_function)(layer)
    layer = Dense(int(nodes / 8), activation=user_defined_activation_function)(layer)
    layer = Dense(int(nodes / 16), activation=user_defined_activation_function)(layer)
    output_1 = Dense(1, activation=user_defined_activation_function)(layer)

    # Create the model composed of the defined layers and nodes
    model = Model(inputs=Input_1, outputs=output_1)

    '''
     - DEFINE OPTIMIZER, LEARNING RATE, AND LOSS FUNCTIONS -
    '''

    '''
    OPTIMIZATION FUNCTION
    '''

    # Capture user defined optimization function if different from default Adam
    user_defined_optimization_function = NeuralNetworkDict().OptimizationFunction(optimization_function)

    '''
    LEARNING RATE
    '''
    # Check if the learning_rate is None. If so, grab default learning rate for specified optimization function from LearninRate function/dictionary
    # If learning rate is provided by user, use that value instead
    if not learning_rate:
        dict_learning_rate = NeuralNetworkDict().LearningRate(user_defined_optimization_function)
        logger.info(f'Learning rate not provided. Using optimization function {user_defined_optimization_function} with default learning rate: {dict_learning_rate}')
        # For future reference:
        # We can pass getattr the following optimization parameter settings in order {'lr', 'beta_1', 'beta_2', 'epsilon', 'schedule_decay'}
        # Here we are setting optimizer to the equivalent of keras.optimizers.opt(learning_rate = some_value)
        # Due to the user optimization function being a string from a dictionary key, we use getattr to do the equivalent call
        optimizer = getattr(keras.optimizers, str(user_defined_optimization_function))(dict_learning_rate)
    else:
        logger.info(f'Learning rate provided is: {learning_rate}')
        optimizer = getattr(keras.optimizers, str(user_defined_optimization_function))(learning_rate)

    '''
    LOSS FUNCTION
    '''
    # Capture user defined loss function if different from default MSE
    loss_function = loss_function
    user_defined_loss_function = NeuralNetworkDict().LossFunction(loss_function)

    # Log all of the functions being used for training
    logger.info(f'    Optimization Function: {user_defined_optimization_function}')
    logger.info(f'      Activation Function: {user_defined_activation_function}')
    logger.info(f'            Loss Function: {user_defined_loss_function}')

    # Compile the model with user define optimization and loss functions. Capturing metrics for plotting purposes
    model.compile(optimizer=optimizer, loss=user_defined_loss_function, metrics=['mae', 'acc', 'mse'])

    # Print a summary of the neural network architecture
    logger.info("#=============================== MODEL SUMMARY ===============================#")
    model.summary(print_fn=logger.info)
    logger.info("Optimization Configuration:")
    logger.info(model.optimizer.get_config())
    logger.info(optimizer)

    # Print configuration details of neural network layer 3 for debugging purposes.
    logger.info("\n")
    logger.info("Layer 3 Configuration:")
    layer3 = model.layers[3].get_config()
    logger.info(layer3)

    logger.info(
        '#=================================== BEGIN TRAINING NEURAL NETWORK ===================================#\n')

    history = model.fit(Scaled_X_train, Scaled_y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(Scaled_X_test, Scaled_y_test))

    logger.info("Done Training Neural Network. Saving model to HDF5 file")
    logger.info("Calculating Predictions and Un-scaling Data...")

    # Generate predictions to plot against truth values with testing set
    Scaled_y_Predictions = model.predict(Scaled_X_test, verbose=1)

    # Due to different possible scaler types options, a conditional statement is performed to properly unscale predictions
    if user_defined_scaler == 'StandardScaler':
        Unscaled_y_Predictions = scaler.inverse_transform(Scaled_y_Predictions)
    else:
        Unscaled_y_Predictions = scalerY.inverse_transform(Scaled_y_Predictions)

    model.save("SurrogateModel_%dLayers_%dNodes_%dEpochs.h5" % (layers, nodes, epochs))
    tf.keras.utils.plot_model(model, to_file=f'Neural_Network_Model_{layers}Layers_{nodes}Nodes_{epochs}Epochs.png', show_shapes=True, show_layer_names=True)

    # Returning variables which are used for plotting purposes
    return history, y_test, Unscaled_y_Predictions, output_col_name, y_output_data, df, df_concat


if __name__ == '__main__':

    # Create a unique directory to store results
    current_dir = GenDirStructure()

    # Capture neural network parameter outputs into a log file
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    datetime_FileName = 'Train_Neural_Network_' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + '.log'
    fh = logging.FileHandler(filename=datetime_FileName, mode='w')
    logger.addHandler(fh)
    logger.info("\n")
    logger.info(
        '#=============================== DEEP FEED FORWARD NEURAL NETWORK LOG ===============================#\n')
    logger.info('Built with Python 3.6')
    logger.info('Machine Learning Packages: Keras, SciKit-Learn, Tensorflow (backend GPU processing)')
    logger.info('Python libraries: Numpy, SciPy, Pandas, Matploblib, Seaborn\n')

    # Capture user passed arguments
    parser = create_parser()
    args = parser.parse_args()
    print(parser.format_values())

    input_csv_file = args.input_file

    num_training_columns = args.training_columns
    test_split_test_size = args.test_size
    scaling_type = args.scaling_type
    
    batch_size = args.batch_size
    num_layers = args.layers
    num_nodes = args.nodes
    num_epochs = args.epochs

    activation_function = args.activation_function
    optimization_function = args.optimization_function
    learning_rate = args.learning_rate
    loss_function = args.loss_function

    # Train the neural network
    history, y_test, Unscaled_y_Predictions, output_col_name, y_output_data, df , df_concat = train_neural_network(input_csv_file, num_training_columns, test_split_test_size, scaling_type, batch_size, num_layers, num_nodes, num_epochs, activation_function, optimization_function, learning_rate, loss_function)

    # Generate post-processing plots
    # logging.info("\n################################# Post-Processing Plots #################################")
    logging.getLogger('matplotlib').setLevel(logging.WARNING)  # Only print out warning messages from Matplotlib

    # Capture plotting function class into plots variable
    plots = PlottingFunctions()
    logger.info('\n#===================================== GENERATING PLOTS =============================================#\n')
    plots.Histogram(output_col_name, y_output_data)
    logger.info("Histogram Plot Generated")

    plots.CorrHeatMap(df_concat)
    logger.info("Correlation HeatMap Plot Generated")

    plots.LossPlot(history)
    logger.info("Loss Plot Generated")

    plots.MSEPlot(history)
    logger.info("MSE Plot Generated")

    plots.MAEPlot(history)
    logger.info("MAE Plot Generated")

    plots.LogLogPlot(y_test, Unscaled_y_Predictions, output_col_name)
    logger.info("Log-Log Plot Generated\n")

    plots.ScatterPlot(y_test, Unscaled_y_Predictions, output_col_name)
    logger.info("Scatter Plot Generated\n")

    plots.ResidualPlot(y_test, Unscaled_y_Predictions)
    logger.info("Residual Plot Generated")
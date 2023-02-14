# Feed-Forward-Neural-Network
A generalized script to train a neural network

> NOTE: Python 3.9.5 was used for testing

### 1. Create a virtual environment called machine_learning
```
> python -m venv machine_learning
```

### 2. Activate the virtual environment (Windows OS)
```
> machine_learning\Scripts\activate
```

### 3. Install initial requirements with pip
Nagivate to the root directory of this repository. It should hold the requirements.txt file. Run the following command:

```
> pip install -r requirements.txt
```

## Running Demo

### Run Python Script
```
> \path-to-repo\feed_forward_neural_network\ffnn_train.py
```
The **ffnn_config.txt** file will automatically be read and load the *Tetuan_City_Power_Consumption.csv* file by default. 

## Running New Data
In order to run on a new dataset, users will need to update the **ffnn_config.txt** file. In particular, modify the following parameters,

* --input_file : Should point to the path to the new CSV data
* --training_columns : This value should match the number of input features in the new CSV data. 

> NOTE: By default, the last column in the CSV file is taken as the single output column. The neural network will attempt to generate predictions to these outputs based on any arbitrary new input data.
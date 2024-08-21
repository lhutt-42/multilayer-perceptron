<h1 align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="../assets/banner/mlp-dark.svg"
    >
    <img
      alt="multilayer-perceptron"
      src="../assets/banner/mlp-light.svg"
      width="500"
    >
  </picture>
  <br>
  <small>Installation, Configuration & Usage Guide</small>
</h1>


This guide provides step-by-step instructions for installing, configuring, and using the multilayer-perceptron.\
Follow these steps to set up your environment, customize your model, and run training and prediction tasks.


## Installation

Before you can run the multilayer perceptron (MLP) model, ensure that you have all the necessary dependencies installed.
Follow the steps below to set up your environment:


### 1. Clone the Repository:

```bash
git clone https://github.com/lucas-ht/multilayer-perceptron.git
cd multilayer-perceptron
```


### 2. Set Up a Virtual Environment:

```bash
python3 -m venv venv
source venv/bin/activate
```


### 3. Install the Required Packages:

Ensure that you have all necessary Python packages installed.
You can install them via pip:

```bash
pip install -r requirements.txt
```


## Configuration

The configuration of the MLP model is done using a `.pkl` file, which defines the architecture and settings of the model.
Below is a detailed breakdown of the configuration file format.


### Structure of the Configuration File

The `.pkl` file contains the following sections:

* Model Configuration: Defines the overall structure of the model, including the number of epochs, optimizer settings, and early stopping criteria.

* Layer Definitions: Specifies the architecture of the MLP, including the number of layers, layer sizes, activation functions, weight initializers, bias initializers, and regularizers.


### Example Configuration File

Here’s an example configuration file:

```pkl
amends "pkl/AppConfig.pkl"

import "pkl/model/Activations.pkl"
import "pkl/model/Initializers.pkl"
import "pkl/model/Layers.pkl"
import "pkl/model/Models.pkl"
import "pkl/model/Optimizers.pkl"
import "pkl/model/Regularizers.pkl"
import "pkl/model/Training.pkl"

model = new Models.BatchModel {
    epochs = 15000
    optimizer = new Optimizers.AdamOptimizer {
        learning_rate = 0.001
        beta1 = 0.9
        beta2 = 0.999
    }
    early_stopping = new Training.EarlyStopping {
        patience = 1000
        delta = 0.0
    }
    layers {
        new Layers.DenseLayer {
            layer_size = "input"
            activation = new Activations.SigmoidActivation {}
            weight_initializer = new Initializers.XavierInitializer {}
            bias_initializer = new Initializers.ZeroInitializer {}
            regularizer = new Regularizers.L1Regularizer {}
        }
        new Layers.DenseLayer {
            layer_size = 18
            activation = new Activations.SigmoidActivation {}
            weight_initializer = new Initializers.XavierInitializer {}
            bias_initializer = new Initializers.ZeroInitializer {}
            regularizer = new Regularizers.L2Regularizer {}
        }
        new Layers.DenseLayer {
            layer_size = 12
            activation = new Activations.SigmoidActivation {}
            weight_initializer = new Initializers.XavierInitializer {}
            bias_initializer = new Initializers.ZeroInitializer {}
            regularizer = new Regularizers.L2Regularizer {}
        }
        new Layers.DenseLayer {
            layer_size = 6
            activation = new Activations.SigmoidActivation {}
            weight_initializer = new Initializers.XavierInitializer {}
            bias_initializer = new Initializers.ZeroInitializer {}
            regularizer = new Regularizers.L2Regularizer {}
        }
        new Layers.DenseLayer {
            layer_size = "output"
            activation = new Activations.SoftmaxActivation {}
            weight_initializer = new Initializers.HeInitializer {}
            bias_initializer = new Initializers.ZeroInitializer {}
            regularizer = null
        }
    }
}
```


## Usage

Once your environment is set up and your configuration file is ready, you can start training the MLP model.


### Splitting the dataset

Before training, you need to split your dataset into training and testing sets.
Use the split command to accomplish this:

```bash
python mlp.py --seed <seed> split <dataset> --test-size <ratio> --out-dir <directory>
```

* `--seed`: Set the seed for reproducibility.

* `dataset`: Path to the dataset.

* `--test-size`: Ratio of the dataset to use for testing (e.g., 0.1 for 90% training data and 10% testing data).

* `--out-dir`: Path where the training set will be saved.


### Training

Once the dataset is split, you can train the model using the train command:

```bash
python mlp.py --seed <seed> train <dataset> <model> --out-dir <directory>
```

* `--seed`: Set the seed for reproducibility.

* `dataset`: Path to the training dataset.

* `model`: Path to the model configuration file.

* `--out-dir`: Path where the model and metrics will be saved.

During training, the script will output relevant metrics such as loss and accuracy to monitor progress.\
If early stopping is configured, the training will stop when the improvement in validation loss is below the specified threshold for a set number of epochs.\
At the end of the training, plots will be displayed showing the progress over epochs.


### Making Predictions

After training, use the predict command to make predictions on new data:

```bash
python mlp.py --seed <seed> train <dataset> <model>
```

* `--seed`: Set the seed for reproducibility.

* `dataset`: Path to the testing dataset.

* `model`: Path to the model configuration file.

This script will load the trained model, run predictions on the test data, and display the loss and accuracy of the model.

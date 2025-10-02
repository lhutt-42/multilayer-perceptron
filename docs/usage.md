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


This guide provides step-by-step instructions for installing, configuring, and using the multilayer-perceptron using modern Python tooling.


## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) - A fast Python package installer and resolver (recommended)
- Git


### Quick Setup with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a blazingly fast Python package installer and resolver written in Rust. It's 10-100x faster than pip and provides better dependency resolution.

#### 1. Install uv

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

#### 2. Clone and Setup

```bash
git clone https://github.com/lucas-ht/multilayer-perceptron.git
cd multilayer-perceptron
uv sync
```

That's it! All dependencies will be installed in a managed virtual environment.


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

model = new Models.MiniBatchModel {
    batch_size = 16
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
            optimizer = new Optimizers.GradientDescentOptimizer {
                learning_rate = 0.001
            }
        }
        new Layers.DenseLayer {
            layer_size = 18
            activation = new Activations.ReluActivation {}
            weight_initializer = new Initializers.XavierInitializer {}
            bias_initializer = new Initializers.RandomInitializer {}
            regularizer = new Regularizers.L2Regularizer {}
        }
        new Layers.DenseLayer {
            layer_size = "output"
            activation = new Activations.SoftmaxActivation {}
            weight_initializer = new Initializers.HeInitializer {}
            bias_initializer = new Initializers.ZeroInitializer {}
            regularizer = null
            gradient_clipping = 2.0
        }
    }
}
```

> [!WARNING]
The example configuration above is intended to demonstrate various options and may not perform optimally.


## Usage

Once your environment is set up and your configuration file is ready, you can start using the MLP.

> **Note**: If using `uv`, prefix all commands with `uv run`. If using a traditional virtual environment, activate it first with `source venv/bin/activate`.


### Global Options

```bash
# With uv
uv run mlp.py [--seed <seed_value>] <command>
```

* `--seed`: Set the seed value for all random number generators. Ensures reproducible results.


### Splitting the Dataset

Before training, split your dataset into training and testing sets:

```bash
# With uv
uv run mlp.py split <dataset> [--test-size <ratio>] [--out-dir <directory>]
```

**Arguments:**
* `dataset`: Path to the dataset CSV file
* `--test-size`: Test set ratio (default: 0.2 for 80/20 split)
* `--out-dir`: Output directory (default: `./data`)

**Example:**
```bash
uv run mlp.py split data/data.csv --test-size 0.2 --out-dir ./data
```


### Training

Train the model with your configuration:

```bash
# With uv
uv run mlp.py train <dataset> <model> [options]

```

**Arguments:**
* `dataset`: Path to the training dataset
* `model`: Path to the model configuration `.pkl` file

**Options:**
* `--out-dir <directory>`: Where to save the trained model (default: `./models`)
* `--no-plot`: Skip displaying plots after training
* `--plot-n <n>`: Number of past training runs to overlay in plots (default: 2)
* `--plot-raw`: Display raw metrics instead of smoothed curves

**Example:**
```bash
uv run mlp.py train ./data/train.csv ./model.test.pkl --plot-raw
```

**Training Output:**

During training, you'll see real-time metrics:
```
INFO: epoch 0 - train loss: 0.655 - test loss: 0.656 - train accuracy: 0.633 - test accuracy: 0.631
```

After training completes (unless `--no-plot` is used), a visualization window displays:
- **Grid layout**: 2×3 subplots showing all metrics
- **Color-coded**: Each metric has a unique color palette
- **Metrics tracked**: Loss, Accuracy, Precision, Recall, F1 Score
- **Train/Test split**: Each metric shows both training and validation curves


### Making Predictions

Evaluate the trained model on test data:

```bash
# With uv
uv run mlp.py predict <dataset> <model>
```

**Arguments:**
* `dataset`: Path to the test dataset
* `model`: Path to the trained model JSON file (saved after training)

**Example:**
```bash
uv run mlp.py predict ./data/test.csv ./models/model.json
```

**Output:**
```
INFO: Model Loss: 0.6117
INFO: Model Accuracy: 0.5789
INFO: Model Precision: 0.0000
```

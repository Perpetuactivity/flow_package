# Network Flow Classification Package

## About

This package provides tools for network traffic flow classification and analysis. It enables users to identify, categorize, and analyze network traffic patterns based on various features and characteristics of network flows.

## Use Cases

- **Network Security Monitoring**: Detect anomalous traffic patterns and potential security threats by classifying network flows
- **Traffic Engineering**: Analyze network traffic to optimize routing and resource allocation
- **Application Performance Analysis**: Identify and classify application-specific traffic for performance tuning
- **Quality of Service Implementation**: Categorize traffic flows to apply appropriate QoS policies
- **Behavioral Analytics**: Study traffic patterns to understand user and application behaviors

## Features

- Flow extraction from packet capture files
- Feature computation for network flows
- Classification of flows using machine learning algorithms
- Visualization of flow characteristics
- Export capabilities for further analysis

## Getting Started

### Installation

```bash
pip install flow-package
uv add flow-package
```

### Basic Usage

This package provides tools for preprocessing network flow data and creating reinforcement learning environments for flow classification.

#### Data Preprocessing

> [!CAUTION]
> Path: Please use **ABSOLUTE PATH**.

```python
from flow_package import data_preprocessing

# Preprocess data with categorical features
train_data, test_data, label_list = data_preprocessing(
    train_data="path/to/training_data.csv", 
    test_data="path/to/test_data.csv",
    categorical_index=["Protocol", "Destination Port"]
)

# Alternatively, split a single dataset into train/test
train_data, test_data, label_list = data_preprocessing(
    train_data="path/to/dataset.csv",
    categorical_index=["Protocol", "Destination Port"]
)
```

| argument name | require | detail |
| ------------- | :-----: | ------ |
| train_data    |    x    | input csv file as training data <br>(or both train and test data) |
|  test_data    |         | input csv file as testing data  |
| categorical_index | x   | choose column name of categorical |

#### Binary Flow Classification

For binary classification of network flows (normal vs attack):

```python
from flow_package import BinaryFlowEnv, InputType
import numpy as np

# Prepare input for the environment
input_type = InputType(
    input_features=train_data.drop(columns=["Number Label"]),
    input_labels=train_data["Number Label"],
    reward_list=[1.0, -1.0]  # Reward for correct and incorrect classifications
)

# Create the environment
env = BinaryFlowEnv(input_type)

# Use the environment for reinforcement learning
observation = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Replace with your agent's action
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated:
        observation = env.reset()
```

#### Multi-class Flow Classification

For classifying flows into multiple categories:

```python
from flow_package import MultipleFlowEnv, InputType
import numpy as np

# Create reward matrix for multi-class classification
num_classes = len(label_list)
reward_matrix = np.ones((num_classes, num_classes)) * -1.0
np.fill_diagonal(reward_matrix, 1.0)

# Prepare input for the environment
input_type = InputType(
    input_features=train_data.drop(columns=["Number Label"]),
    input_labels=train_data["Number Label"],
    reward_list=reward_matrix
)

# Create the environment
env = MultipleFlowEnv(input_type)

# Use the environment with your reinforcement learning algorithms
```

For more detailed examples and advanced usage, check out the documentation.

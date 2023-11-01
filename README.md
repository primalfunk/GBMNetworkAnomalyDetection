# GBM Network Anomaly Detection

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Generation](#data-generation)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains a Python project focused on detecting network anomalies using a Gradient Boosting Machine (GBM). The project aims to provide a sophisticated model trained on simulated but realistic network data, generated using the Faker library. The model is trained to identify anomalous network behavior and can be particularly useful for security analytics.

## Features

- Data generation using Faker to simulate network logs with nuanced patterns
- Data preprocessing including handling of missing values and normalization
- Feature engineering to extract meaningful information from raw data
- Model training using a Gradient Boosting Machine
- Hyperparameter tuning using GridSearch
- Model evaluation including a feature importance plot and a confusion matrix

## Installation

To get started, clone the repository and install the requirements:

\```bash
git clone https://github.com/primalfunk/GBMNetworkAnomalyDetection.git
cd GBMNetworkAnomalyDetection
pip install -r requirements.txt
\```

## Usage

### Data Generation

Run `data_generator.py` to generate simulated network log data.

\```bash
python data_generator.py
\```

This will create a dataset with various features like source and destination IPs, ports, and protocols.

### Data Preprocessing

Run `data_pre_processor.py` to clean and preprocess the generated data.

\```bash
python data_pre_processor.py
\```

This step includes encoding categorical features and normalizing numerical features.

### Feature Engineering

Run `feature_engineering.py` to create additional features for model training.

\```bash
python feature_engineering.py
\```

This includes extracting year, month, day, and hour from the timestamp data.

### Model Training

Run `model.py` to train the GBM model.

\```bash
python model.py
\```

The model will be trained and saved in a serialized format.

### Model Evaluation

Run `model_evaluation.py` to generate evaluation plots.

\```bash
python model_evaluation.py
\```

This will generate a feature importance plot and a confusion matrix to evaluate the model's performance.

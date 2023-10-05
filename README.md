# train-with-fsdp

Code used to fine-tune this model: [abacaj/starcoderbase-1b-sft](https://huggingface.co/abacaj/starcoderbase-1b-sft).

Note the data in folder `data/` is not the full training data used. You can find the full set here: [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1)

# How to run

Install dependencies:
```
python -m venv env \
  && source env/bin/activate \
  && pip install -r requirements.txt
```

Run training code:
```
torchrun --nnodes=1 --nproc-per-node=<REPLACE_WITH_NUMBER_OF_GPUS> train.py
```

To add data place jsonl files in data/ and edit `train.py` line `:154`, `:155`.

# Benchmarks
![image](https://github.com/abacaj/train-with-fsdp/assets/7272343/9c299936-c261-4992-b6d1-d61b0d6da15e)

# Charts
![image](https://github.com/abacaj/train-with-fsdp/assets/7272343/eab7e07a-f8ca-4ee3-8b33-b6e7a4016d18)

See: [wandb](https://api.wandb.ai/links/abacaj1/c4nkcs9r)

# Overview

Welcome to our latest project, a comprehensive toolkit for training language models using the Fully Sharded Data Parallel (FSDP) technique. This project provides a robust and efficient way to train language models on large datasets, with a focus on performance and scalability. The codebase includes scripts for setting up the model, logging statistics, saving the model, preprocessing and tokenizing data, and handling the data. It also includes various utility functions related to distributed training using PyTorch.

# Technologies and Frameworks

This project is built with Python and uses several popular libraries and frameworks:

- **PyTorch**: An open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.
- **Transformers**: A state-of-the-art library for Natural Language Processing (NLP) and transfer learning.
- **wandb**: A tool for machine learning experiment tracking, dataset versioning, and model management.

The project also includes a `requirements.txt` file that lists all the necessary software packages and their versions, including `appdirs`, `certifi`, `click`, `numpy`, `torch`, `transformers`, and `wandb`.

# Installation

Follow these steps to install and start working with the project:

## Step 1: Clone the Repository

First, clone the repository to your local machine. Open your terminal and run the following command:

```bash
git clone https://github.com/abacaj/train-with-fsdp.git
```

## Step 2: Navigate to the Project Directory

Navigate to the project directory using the following command:

```bash
cd train-with-fsdp
```

## Step 3: Install Python Packages

The project requires several Python packages. You can install these packages using pip. Run the following command:

```bash
pip install -r requirements.txt
```

This command installs all the packages listed in the `requirements.txt` file.


## Step 4: Verify Installation

After installing all the required packages, verify the installation by running the Python scripts in the project. For example, you can run the `train.py` script as follows:

```bash
python train.py
```

If the script runs without any errors, the installation is successful. You can now start working with the project.

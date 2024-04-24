# TNSDC---Generative-AI

# GAN (Generative Adversarial Network) for Fashion MNIST Dataset

This repository contains a simple implementation of a Generative Adversarial Network (GAN) using TensorFlow 2.x for generating fashion items resembling the Fashion MNIST dataset. GANs are a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a game.

## Dataset

The model is trained on the Fashion MNIST dataset, which consists of 60,000 training images and 10,000 testing images. The images are grayscale and have a resolution of 28x28 pixels. The dataset includes 10 different fashion items: T-shirts/tops, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.

## Model Architecture

### Generator
- Dense layer: 7x7x64 units with ReLU activation
- Reshape layer: Reshapes the output to (7, 7, 64)
- Conv2DTranspose layers: 
  - Filters: 64, kernel size: 3x3, strides: 2x2, activation: ReLU
  - Filters: 32, kernel size: 3x3, strides: 2x2, activation: ReLU
  - Filters: 1, kernel size: 3x3, strides: 1x1, activation: Sigmoid

### Discriminator
- Input layer: Input shape of (28, 28, 1)
- Conv2D layers:
  - Filters: 32, kernel size: 3x3, strides: 2x2, activation: ReLU
  - Filters: 64, kernel size: 3x3, strides: 2x2, activation: ReLU
- Flatten layer
- Dense layer: 1 unit, no activation function

### Optimizers
- Generator: Adam optimizer with a learning rate of 0.001 and beta_1 of 0.5
- Discriminator: RMSprop optimizer with a learning rate of 0.005

## Training

The model is trained for 50 epochs with a batch size of 512. During training, both the generator and discriminator are updated alternatively to optimize their respective objectives.

## Results

After training, the GAN should be able to generate new fashion items that resemble the ones in the Fashion MNIST dataset. You can visualize the generated samples after each epoch to monitor the training progress.

## Requirements

- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm
- pandas

## Usage

You can run the code in a Jupyter notebook or any Python environment that supports TensorFlow 2.x. Make sure to install the required packages mentioned above before running the code.


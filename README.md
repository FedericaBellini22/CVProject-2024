# CVProject-2024

# Parking Lot Occupancy Detection Using CNN

This project is a deep learning-based approach to detecting parking lot occupancy using images from the [PKLot dataset](https://web.inf.ufpr.br/vri/databases/pklot/). The model is implemented using TensorFlow and Keras, with a Convolutional Neural Network (CNN) architecture. The project includes preprocessing the dataset, training the CNN model, and evaluating its performance with metrics such as accuracy, loss, and confusion matrix.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Confusion Matrix](#confusion-matrix)
- [Literature Reference](#literature-reference)
- [Results](#results)
- [License](#license)

## Project Overview

The goal of this project is to classify parking lot images into two categories:
- **Empty**: The parking space is unoccupied.
- **Occupied**: The parking space is occupied by a vehicle.

The project leverages a Convolutional Neural Network (CNN) to process the input images, and the model achieves high accuracy on the test dataset.

This work is inspired by and is a re-implementation of methods discussed in the article *"Real-time image-based parking occupancy detection using deep learning"*.

## Dataset

The dataset used for this project is the [PKLot dataset](https://web.inf.ufpr.br/vri/databases/pklot/), a public dataset containing images of parking lots captured under different weather conditions and angles. It has two classes:
- **Empty**: Parking spaces without cars.
- **Occupied**: Parking spaces with cars.

### Dataset Structure

After extraction and organization, the dataset is split into three sets:
- **Training Set**: 70% of the images
- **Validation Set**: 20% of the images
- **Test Set**: 10% of the images

## Requirements

To run this project, you'll need the following libraries and dependencies:

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn

You can install the required libraries using the following command:

```bash
pip install tensorflow numpy matplotlib scikit-learn


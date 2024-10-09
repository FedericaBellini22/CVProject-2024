# CVProject-2024

# Parking Lot Occupancy Detection Using CNN

This project is a deep learning-based approach to detecting parking lot occupancy using images from the [PKLot dataset](https://web.inf.ufpr.br/vri/databases/parking-lot-database/). The model is implemented using TensorFlow and Keras, with a Convolutional Neural Network (CNN) architecture. The project includes preprocessing the dataset, training the CNN model, and evaluating its performance with metrics such as accuracy, loss, and confusion matrix.

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

The dataset used for this project is the [PKLot dataset](https://web.inf.ufpr.br/vri/databases/parking-lot-database/), a public dataset containing images of parking lots captured under different weather conditions and angles. It has two classes:
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
```

## Installation

1) Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/parking-lot-occupancy-detection.git
```

2) Navigate to the project directory:

```bash
cd parking-lot-occupancy-detection
```
    
3) Install the required dependencies:

```bash
pip install -r requirements.txt
```

4) Download and extract the PKLot dataset to the appropriate directory. (https://web.inf.ufpr.br/vri/databases/parking-lot-database/)

## Model Architecture


# CNN Model Description

## Architecture Overview
This CNN (Convolutional Neural Network) model is designed for binary classification tasks, where the goal is to distinguish between two classes. The architecture consists of three convolutional layers followed by max-pooling, flattening, and dense layers. Here's a detailed breakdown of the layers and their roles:

1. **Input Layer**: 
   - Input shape: `(IMG_HEIGHT, IMG_WIDTH, 3)` representing the height, width, and 3 color channels (RGB) of the input images.

2. **Convolutional Layer 1**:
   - `Conv2D(32, (3, 3), activation='relu')`: Applies 32 filters of size 3x3 to the input image.
   - **Activation**: ReLU (Rectified Linear Unit) introduces non-linearity, helping the model learn complex patterns.
   - **Output shape**: `(None, 126, 126, 32)` - 32 feature maps.

3. **Max Pooling Layer 1**:
   - `MaxPooling2D((2, 2))`: Reduces the spatial dimensions by taking the maximum value over a 2x2 window.
   - **Output shape**: `(None, 63, 63, 32)`.

4. **Convolutional Layer 2**:
   - `Conv2D(64, (3, 3), activation='relu')`: Uses 64 filters of size 3x3.
   - **Output shape**: `(None, 61, 61, 64)`.

5. **Max Pooling Layer 2**:
   - `MaxPooling2D((2, 2))`: Further reduces spatial dimensions.
   - **Output shape**: `(None, 30, 30, 64)`.

6. **Convolutional Layer 3**:
   - `Conv2D(64, (3, 3), activation='relu')`: Another layer with 64 filters of size 3x3.
   - **Output shape**: `(None, 28, 28, 64)`.

7. **Flatten Layer**:
   - `Flatten()`: Flattens the 3D output of the last convolutional layer into a 1D vector.
   - **Output shape**: `(None, 50176)` - a single vector that can be fed into the dense layers.

8. **Dense Layer 1**:
   - `Dense(64, activation='relu')`: Fully connected layer with 64 neurons.
   - **Activation**: ReLU.
   - Helps learn complex representations from the extracted features.
   - **Output shape**: `(None, 64)`.

9. **Output Layer**:
   - `Dense(1, activation='sigmoid')`: Final layer with a single neuron.
   - **Activation**: Sigmoid function, which outputs a probability between 0 and 1 for binary classification.
   - **Output shape**: `(None, 1)`.

## Compilation
The model is compiled with:
- **Optimizer**: `adam` - adaptive learning rate optimization algorithm, efficient and widely used for training deep learning models.
- **Loss Function**: `binary_crossentropy` - suitable for binary classification problems.
- **Metrics**: `accuracy` - to monitor the accuracy during training and evaluation.

This architecture is effective for extracting spatial features through convolutional layers, reducing dimensions through pooling layers, and learning complex relationships through dense layers. The use of a sigmoid activation in the output layer makes it suitable for tasks such as classifying whether an image belongs to one category or another.


## Model Summary

| Layer (type)         | Output Shape        | Param #   |
|----------------------|---------------------|-----------|
| conv2d (Conv2D)      | (None, 126, 126, 32) | 896       |
| max_pooling2d (MaxPooling2D) | (None, 63, 63, 32) | 0     |
| conv2d_1 (Conv2D)    | (None, 61, 61, 64)   | 18,496    |
| max_pooling2d_1 (MaxPooling2D) | (None, 30, 30, 64) | 0 |
| conv2d_2 (Conv2D)    | (None, 28, 28, 64)   | 36,928    |
| flatten (Flatten)   | (None, 50176)        | 0         |
| dense (Dense)       | (None, 64)           | 3,211,328 |
| dense_1 (Dense)     | (None, 1)            | 65        |

**Total params:** 3,267,713 (12.47 MB)  
**Trainable params:** 3,267,713 (12.47 MB)  
**Non-trainable params:** 0 (0.00 B)
## Training the Model

To train the model, open the Jupyter notebook and run all the cells:

1) Ensure that you have Jupyter installed. If not, install it using:

   ```bash
   pip install jupyter
   ```
2) Open the Jupyter notebook:
   
   ```bash
   jupyter notebook parking-occupancy-detection_cnn.ipynb
   ```
3) Run all the cells to:

  - Preprocess the dataset
  - Build the CNN model
  - Train the model using the PKLot dataset

![Senza titolo](https://github.com/user-attachments/assets/c402aa84-1a84-44bd-85f0-35ac4627988a)



## Evaluating the Model

After training, you can evaluate the model on the test set using:
  ```bash
  # Evaluate on the test dataset
  results = model.evaluate(test_dataset)
  print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")
  ```
(Already in parking-occupancy-detection_cnn.ipynb code)

## Confusion Matrix 

To visualize how well the model performed for each class (empty or occupied), you can plot a confusion matrix:

![Immagine WhatsApp 2024-10-06 ore 22 55 44_802c38dc](https://github.com/user-attachments/assets/2bf9cf47-4ae8-4766-893e-1db95db2dff6)


## Literature Reference

This project is inspired by the work presented in the article:

    Real-time image-based parking occupancy detection using deep learning
    Available at CEUR-WS.org, Vol-2087

This article presents a method for real-time parking occupancy detection using deep learning techniques, and I have adapted and re-implemented some of the key methods from the paper in this project.

## Results

  - Test Loss: 0.015
  - Test Accuracy: 99.75%

The model performs extremely well, with an accuracy of 99.75% on the test dataset. The confusion matrix and other metrics show that the model is highly reliable in distinguishing between empty and occupied parking spaces.

## License

### How to Use It:
- Copy the entire content of the above Markdown into your repository’s `README.md` file.
- Customize the GitHub repository URL (`git clone https://github.com/your-username/parking-lot-occupancy-detection.git`) and paths as necessary.
- Add the correct `requirements.txt` if applicable to your project.

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

The model is a Convolutional Neural Network (CNN) with the following architecture:
- **Input Layer**: Takes an input image of size 128x128x3 (RGB).
- **3 Convolutional Layers**: Extract spatial features from the images using 32, 64, and 128 filters respectively.
- **MaxPooling Layers**: Reduce the spatial dimensions of the feature maps.
- **Fully Connected Layer**: A dense layer with 128 units.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification (empty or occupied).

## Model Summary

| Layer (type)                 | Output Shape         | Param #     |
|------------------------------|----------------------|-------------|
| conv2d (Conv2D)              | (None, 126, 126, 32) | 896         |
| max_pooling2d (MaxPooling2D) | (None, 63, 63, 32)   | 0           |
| conv2d_1 (Conv2D)            | (None, 61, 61, 64)   | 18,496      |
| max_pooling2d_1 (MaxPooling2D)| (None, 30, 30, 64)  | 0           |
| conv2d_2 (Conv2D)            | (None, 28, 28, 128)  | 73,856      |
| max_pooling2d_2 (MaxPooling2D)| (None, 14, 14, 128) | 0           |
| flatten (Flatten)            | (None, 25088)        | 0           |
| dense (Dense)                | (None, 128)          | 3,211,392   |
| dense_1 (Dense)              | (None, 1)            | 129         |
| **Total params**             |                      | **3,304,769**|
| **Trainable params**         |                      | **3,304,769**|
| **Non-trainable params**     |                      | **0**       |

Total parameters: **3,304,769**.

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

![Senza titolo](https://github.com/user-attachments/assets/20618aaf-8d83-493c-83fb-e44b549d2b71)


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

![Senza titolo](https://github.com/user-attachments/assets/3ef7ea06-78e8-4369-a9cc-6cc38a6de3d2)
![Senza titolo-1](https://github.com/user-attachments/assets/dda9d8ef-f7c4-4fe6-b8f9-4ee3addf1050)

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

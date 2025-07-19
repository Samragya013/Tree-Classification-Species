# 🌳 Tree Species Identification - Week-1 Project

This repository contains the source code and documentation for an AI-driven model designed to classify tree species from images. The project leverages deep learning, specifically Convolutional Neural Networks (CNNs), to create a robust classifier that can empower ecological research, conservation efforts, and botanical studies by automating species identification.

## 🚀 Project Overview

The primary objective of this project is to develop a machine learning pipeline capable of accurately identifying 30 different tree species from leaf and tree imagery. Manual identification can be time-consuming and requires specialized botanical knowledge. This tool aims to provide a fast, accessible, and reliable alternative for botanists, environmentalists, and nature enthusiasts alike.

## Key Features

- Classification of 30 distinct tree species.
- Built on a diverse image dataset from Kaggle.
- Initial development and experimentation conducted in Google Colab.
- Foundation for a real-world application in environmental monitoring and education.

## 📊 The Dataset

The model is trained on the Tree Species Identification Dataset available on Kaggle. It is a well-structured dataset ideal for image classification tasks.

- **Source**: [Kaggle – Tree Species Identification Dataset](https://www.kaggle.com)
- **Total Images**: 1,596
- **Total Classes**: 30
- **Data Structure**: Images are organized into separate folders, each named after the corresponding tree species.

### 🔬 Exploratory Data Analysis (EDA)

A preliminary analysis was conducted to understand the dataset's characteristics and inform the preprocessing strategy.

#### Class Distribution

The dataset is mostly balanced, with most classes containing 49–50 images. However, a notable exception is the "other" class, which contains 150 images. This imbalance will be a key consideration during model training to prevent classification bias.

#### Image Properties

An analysis of sample images from each class revealed significant variance in resolution, which is a critical challenge for model input.

- **Channels**: 3 (RGB)
- **Minimum Resolution**: 135 x 146 pixels
- **Maximum Resolution**: 1699 x 1300 pixels
- **Average Resolution**: Approx. 254 x 290 pixels

This heterogeneity necessitates a robust preprocessing pipeline to standardize image dimensions before feeding them into the neural network.

## 🛠️ Technology Stack

This project utilizes a modern stack of Python libraries and tools for machine learning and data science.

- **Environment**: 📓 Google Colab
- **Core Libraries**:
  - **OpenCV (cv2)**: For advanced image processing and manipulation.
  - **NumPy**: For efficient numerical operations and array handling.
  - **Matplotlib**: For data visualization and plotting sample images.
  - **Collections**: For data structure manipulation and analysis.

## 🗓️ Project Roadmap

The project is structured in several phases, from initial exploration to model deployment.

- **Phase 1**: Data Exploration & Analysis
- **Phase 2**: Preprocessing & Augmentation (Image resizing, normalization, and data augmentation)
- **Phase 3**: Model Development (Building and training a CNN-based classification model)
- **Phase 4**: Evaluation & Hyperparameter Tuning (Assessing model performance and optimizing its architecture)
- **Phase 5**: Deployment (Future Scope) (Creating a simple web interface for real-time predictions)

## ✍️ Author

**Samragya Banerjee**

B.Tech in Computer Science & Engineering

A passionate student developer with a keen interest in applying Machine Learning and Computer Vision to address environmental and real-world challenges.

## ⭐ Acknowledgements

Special thanks to Deepak Sir for sharing this valuable dataset with us. Thanks to the Google Colaboratory team for providing a powerful, accessible cloud-based environment for AI development.

# Image Compression
## Introduction
Welcome to the Image Compression project! This project is a simple implementation of image compression using Randomized Singular Value Decomposition (RSVD) and PCA Decomposition techniques. The main objective of this project is to explore and apply dimensionality reduction methods to achieve efficient image compression.

## Motivation
This project was undertaken during my first year of college with the primary goal of gaining hands-on experience in image compression techniques. Additionally, I wanted to delve into the concepts of Singular Value Decomposition (SVD) and Principal Component Analysis (PCA) for their effectiveness in reducing the dimensionality of images while preserving essential visual information.

## Features
Implementation of Randomized Singular Value Decomposition (RSVD) to approximate the SVD of an image matrix efficiently.
Application of PCA Decomposition for dimensionality reduction to obtain the most significant components of the image.
Reconstructed the original codebase using an Object-Oriented Programming (OOP) approach for improved code organization and maintainability.
Optimized the code to enhance computational efficiency, making it suitable for larger images.

## How it Works
The image is first read and represented as a matrix of pixel values.
RSVD is applied to approximate the SVD of the image matrix, retaining the most important singular values and corresponding vectors.
PCA Decomposition is then utilized to reduce the dimensionality of the image while preserving critical visual information.
The compressed image is reconstructed from the reduced representation using the retained components.
The reconstructed image is compared with the original to assess the compression quality.

## Requirements
* Python 3.6 or above
* Numpy
* matplotlib
* scipy
* Pillow

## Usage
* Clone the repository
* run `python3 main.py` in the terminal
  
## Results
### Original Image
![Original Image](lena.jpg)
### Compressed Image
![Compressed Image](rSVD_lena.jpg_k0100.jpg)
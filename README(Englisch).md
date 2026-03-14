Traffic Sign Recognition using Deep Learning

This project implements a Convolutional Neural Network (CNN) for traffic sign classification using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

Requirements

Python 3.12

Required libraries:
- tensorflow
- numpy
- pandas
- opencv-python
- matplotlib
- scikit-learn

Install the required libraries with:

pip install tensorflow numpy pandas opencv-python matplotlib scikit-learn

Dataset

The dataset used in this project can be downloaded from:

https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed

After downloading the dataset, extract it into the following folder:

traffic_sign_project/dataset

Running the model

Run the following command:

python train_model.py

The script will load the dataset, train the CNN model, and evaluate its accuracy.


# MNIST Digit Classification using Deep Learning

## Overview
This project implements a deep learning model to classify handwritten digits from the MNIST dataset. The model is trained using a neural network and tested on unseen images to evaluate its performance. The project also allows users to provide new handwritten digit images for prediction.

## Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels, representing digits from 0 to 9.

## Features
- Preprocessing of MNIST dataset
- Building and training a deep learning model
- Evaluating model accuracy on test data
- Predicting handwritten digits from new images

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-classification.git
   cd mnist-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
### Training the Model
Run the Jupyter notebook:
```bash
jupyter notebook mnist_classification.ipynb
```

### Predicting New Images
To predict a new handwritten digit:
1. Place your image in the project directory.
2. Run the following script:
   ```python
   python predict.py --image new_image.png
   ```

## Results
The trained model achieves an accuracy of **XX%** on the test set.

## Future Enhancements
- Improve model accuracy using CNNs.
- Deploy the model as a web application.
- Integrate with a mobile app for real-time predictions.

## Contributions
Feel free to contribute by raising issues or submitting pull requests.

## License
This project is licensed under the MIT License.


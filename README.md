# SudokuLens - A Real-Time Sudoku Solver

SudokuLens is a real-time Sudoku solver that uses computer vision and a convolutional neural network to detect and solve Sudoku puzzles captured from a live camera feed. It automatically detects a 9x9 Sudoku grid, recognizes handwritten or printed digits, solves the puzzle, and overlays the solution back onto the live video feed. The application is primarily based on Image Processing and Machine Learning (CNN).

## Features

- Detects Sudoku grids in real time using a webcam.
- Recognizes digits using a CNN model trained on the MNIST dataset.
- Solves Sudoku puzzles using a backtracking algorithm.
- Overlays the solution directly onto the original video feed.

## 🛠 Tech Stack

- **Language:** Python  
- **Libraries & Frameworks:**
  - [OpenCV](https://opencv.org/) — for image processing and camera handling
  - [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/) — for building and training the CNN
  - [NumPy](https://numpy.org/) — for numerical operations

## How It Works

1. **Frame Capture:** Captures frames from your webcam using OpenCV.
2. **Grid Detection:** Detects the Sudoku puzzle via contour detection and perspective transformation.
3. **Digit Recognition:** Segments and feeds the cells to the CNN to recognize digits.
4. **Puzzle Solving:** Uses a backtracking algorithm to solve the puzzle.
5. **Overlay Output:** Projects the solution digits back onto the original board in the live video.

## CNN Model Details

- **Training Dataset:** MNIST Handwritten Digits (filtered to include digits 1–9 only)  
- **Input:** 28x28 grayscale images  
- **Architecture:**  
  - 2 Convolutional layers (32 & 64 filters, 3×3 kernel, ReLU)  
  - 1 MaxPooling layer (2×2)  
  - Dropout layers (25% after pooling, 50% before output)  
  - 1 Dense layer (128 units, ReLU)  
  - Output Dense layer (9 units, Softmax for digits 1–9)  
- **Output:** Digit class (1–9)  
- **Model File:** `CNN Digit Training/digitRecognition.h5`  
- **Framework:** Keras with TensorFlow backend

##  Acknowledgements
- MNIST Dataset
- OpenCV and Keras official documentation

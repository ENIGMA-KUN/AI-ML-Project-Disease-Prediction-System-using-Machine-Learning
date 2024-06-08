# Disease Prediction System using Machine Learning

This project provides a comprehensive end-to-end workflow for predicting diseases using machine learning. It includes steps for data preprocessing, model training using various classifiers, evaluation, and prediction. The workflow ensures that the best model is selected based on performance metrics.

## Table of Contents

- [Introduction](#introduction)
- [Setup Instructions](#setup-instructions)
- [Project Structure](#project-structure)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Introduction

The aim of this project is to demonstrate a structured and efficient machine learning workflow for disease prediction. The project covers the following steps:

1. Data preprocessing
2. Splitting the dataset
3. Model training using K-Fold Cross Validation
4. Training different classifiers
5. Evaluating the classifiers
6. Making final predictions

## Setup Instructions

To set up this project on your local machine, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Ensure Graphviz is installed and added to your system PATH**:
    - Download and install Graphviz from [the official website](https://graphviz.gitlab.io/download/).
    - Add the Graphviz `bin` directory to your system PATH.

## Project Structure

.
├── data
│ └── dataset.csv
├── scripts
│ ├── preprocess.py
│ ├── train.py
│ └── predict.py
├── notebooks
│ └── exploration.ipynb
├── output
│ └── machine_learning_workflow.png
├── venv
├── requirements.txt
├── README.md
└── graph.py


- `data/`: Contains the dataset.
- `scripts/`: Contains Python scripts for preprocessing, training, and prediction.
- `notebooks/`: Contains Jupyter notebooks for data exploration.
- `output/`: Contains the output images and results.
- `venv/`: Virtual environment directory.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation file.
- `graph.py`: Script to generate the machine learning workflow diagram.

**Workflow Steps:**

1. **Dataset**: Start with the dataset.
2. **Preprocessing and Splitting**: Split the dataset into training and validation data.
3. **Training Data**: Further split training data into train and test sets.
4. **K-Fold Cross Validation**: Perform K-Fold Cross Validation for model selection.
5. **Train Models**: Train SVM, Naive Bayes, and Random Forest classifiers.
6. **Evaluate Models**: Compute metrics on the test data for each classifier.
7. **Make Predictions**: Make predictions on the validation dataset using the trained models.
8. **Final Prediction**: Combine predictions from all models to make the final prediction.

## Usage

To run the different parts of the workflow, use the following commands:

1. **Preprocess the data**:
    ```bash
    python scripts/preprocess.py
    ```

2. **Train the models**:
    ```bash
    python scripts/train.py
    ```

3. **Make predictions**:
    ```bash
    python scripts/predict.py
    ```

4. **Generate the workflow diagram**:
    ```bash
    python graph.py
    ```

## Dependencies

The project requires the following Python packages:

- graphviz
- matplotlib
- numpy
- pandas
- scikit-learn
- seaborn

You can install all dependencies using:
```bash
pip install -r requirements.txt




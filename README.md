# Deep Learning-Based Skin Cancer Detection Using HAM10000 Dataset

## Project Description
This project implements a deep learning system designed to assist in the preliminary detection of skin cancer by analyzing high-quality images. The system uses the HAM10000 dataset, which is a collection of labeled skin lesion images, for training. The model aims to detect and classify skin cancer lesions into different categories. Users can upload images for analysis, but it is important to note that the system only provides preliminary suggestions, and users should consult a doctor for a confirmed diagnosis.

## Project Objective
- **Objective**: To create a deep learning-based tool for the preliminary detection of skin cancer using the HAM10000 dataset.
- **Goal**: To provide users with an initial analysis of their skin lesions by classifying images into one of the seven categories based on the dataset's labels.

## Dataset
The project uses the **HAM10000** dataset, which contains skin lesion images with associated labels. These labels are used to train a Convolutional Neural Network (CNN) to classify the images into seven different categories. The dataset includes CSV files containing image metadata and image pixel values, which are processed and used to train the model.

**Dataset Files:**
- `hmnist_28_28_RGB.csv`: Image pixel values for images resized to 28x28 in RGB format.
- `hmnist_8_8_RGB.csv`: Image pixel values for images resized to 8x8 in RGB format.
- `HAM10000_metadata.csv`: Metadata of images including lesion attributes such as diagnosis, age, gender, and localization.
- `HAM10000_images_part_1` and `HAM10000_images_part_2`: Image files corresponding to the dataset.

## Model Architecture
The model architecture consists of a **Convolutional Neural Network (CNN)** with multiple convolutional, max-pooling, dropout, and dense layers to perform the classification task. The network is trained using the **Adam optimizer** and the **sparse categorical cross-entropy loss function**. The architecture includes:
- Convolutional layers for feature extraction.
- Max-pooling layers to reduce spatial dimensions.
- Batch normalization for faster convergence.
- Dropout layers to prevent overfitting.
- A final softmax output layer to classify the image into one of seven categories.

## Preprocessing Steps
1. **Data Normalization**: Image pixel values are scaled to the range [0, 1].
2. **Reshaping**: The images are reshaped to a 4D tensor to fit the input requirements of the CNN.
3. **Data Augmentation**: Oversampling is used to handle class imbalance and increase the dataset's robustness.

## Training Strategy
- The model is trained using **K-Fold Cross-Validation** (5 folds), ensuring that the model is tested on different data splits for better generalization.
- **Callbacks** like `ReduceLROnPlateau` (to adjust the learning rate based on validation accuracy) and `EarlyStopping` (to stop training when validation accuracy doesn’t improve) are used to optimize training.

## Example Usage
1. **Training**: The model is trained on the HAM10000 dataset to classify skin lesion images into one of seven categories. The training process includes logging accuracy and loss metrics for each fold of the cross-validation.
2. **Prediction**: Users can upload their own images for the model to classify. The model provides a predicted class, which is displayed with the corresponding label.

## How to Run

1. Clone the repository.
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. Install the required dependencies (if not already installed).

   ```bash
   pip install -r requirements.txt

3. Train the model:
- Load the dataset and train using K-Fold cross-validation.
4. Make predictions:
-  Upload your image and use the classify_image() function to classify it.

  

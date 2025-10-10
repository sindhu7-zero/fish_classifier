# Fish Classification Project

## Project Overview
The Fish Classification Project aims to classify different types of fish using **Convolutional Neural Networks (CNNs)** and **pre-trained models**. The project demonstrates both **CNN from scratch** and **transfer learning** approaches using models like DenseNet, MobileNet, VGG, ResNet, and NASNet.  

The final model is deployed using **Streamlit** for real-time predictions.

---

## Dataset
- **Source**: [Add dataset link here]  
- **Description**: The dataset contains images of multiple fish species with labels.  

**Steps performed**:
1. Downloaded and loaded the dataset.  
2. Applied **data augmentation** to increase dataset diversity.  
3. Preprocessed images: resizing, normalization, and one-hot encoding of labels.

---

## Data Preprocessing & Augmentation
- **Resizing** images to 224x224 pixels  
- **Normalization** of pixel values (0-1)  
- **Data Augmentation**:
  - Rotation
  - Horizontal/Vertical Flip
  - Zoom
  - Width/Height Shifts
  - Brightness Adjustment  

---

## Model Architecture

### CNN from Scratch
- Custom CNN with multiple convolution, pooling, and dense layers  
- ReLU activation and Softmax output  
- Dropout layers for regularization  

### Pre-trained Models (Transfer Learning)
- **DenseNet**, **MobileNet**, **VGG16/VGG19**, **ResNet50/101**, **NASNet**  
- Base layers frozen and new dense layers added for fish classification  
- Fine-tuned on dataset for improved accuracy  

---

## Model Training
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  
- **Epochs**:5 
- **Batch Size**: 32

---

## Results
- Evaluated on test d


# Obj-recognition
Object_Recognition of CIFAR100 using SVM & CNN

# **CIFAR-100 Classification Project**

## **Overview**
This project explores the classification of the **CIFAR-100 dataset** using two primary methodologies:
1. **Convolutional Neural Network (CNN) Approach**
2. **Support Vector Machine (SVM) Approach**

The objective is to compare the performance of CNN and SVM models for **superclasses** and **subclasses** in the CIFAR-100 dataset and analyze the computational cost and accuracy trade-offs.

---

## **Dataset**
The **CIFAR-100 dataset** contains:
- 100 fine-grained **subclasses** grouped into 20 **superclasses**.
- Each image has both a subclass label and a superclass label.

---

## **Key Methodologies**

### **1. CNN Approach (Methodology I)**

#### **1.1 Coarse Model (Superclasses)**
- **Preprocessing:**
  - Normalized image pixel values to [0, 1].
  - Converted labels to categorical format for Keras compatibility.
  - Split data into **training** and **validation** sets.
- **CNN Model Architecture:**
  - **Convolutional Layers:** Extract spatial features from images.
  - **Max Pooling Layers:** Reduce dimensions to prevent overfitting.
  - **Dense Layers:** Fully connected layers for classification.
  - **Activation Function:** ReLU.
- **Optimizer:** Stochastic Gradient Descent (SGD).
- **Results:**  
  - **Accuracy:** 50.24% on the **superclasses**.  
  - Successfully predicted major classes but struggled with some misclassifications due to color similarity (e.g., aquatic mammals vs. vehicles).

#### **1.2 Fine Model (Subclasses)**
- Modified the output layer to predict **100 subclasses**.
- **Results:**  
  - **Accuracy:** 32.94% on **subclasses**.
  - Performance dropped due to the complexity of fine-grained features.

---

### **2. SVM Approach (Methodology II)**
- Implemented **Support Vector Machines (SVMs)** for subclass classification.
- **Linear SVM**: Used to separate classes by finding the optimal hyperplane.
- **Results:**  
  - **Accuracy:** 31.40% on **subclasses**.
  - Achieved similar performance to CNN for subclasses with significantly lower computational cost.

---

## **Model Comparison**

| **Metric**            | **CNN (Superclasses)** | **CNN (Subclasses)** | **SVM (Subclasses)** |
|-----------------------|------------------------|---------------------|---------------------|
| **Accuracy (%)**      | 50.24%                  | 32.94%              | 31.40%              |
| **Computational Cost**| High                    | High                | Low                 |
| **Feature Extraction**| Excellent for coarse    | Challenging for fine| Efficient for fine  |

---

## **Project Structure**

```plaintext
/cifar100-classification
│   README.md                    # Documentation
│   2337862-1.pdf                 # Report file
│   2337862_SVM_.ipynb            # SVM Implementation Notebook
│   2337862_CNN.ipynb             # CNN Implementation Notebook
│
├── data/
│   └── cifar100_train.csv        # Training data
│   └── cifar100_test.csv         # Testing data
├── images/
│   └── confusion_matrix_cnn.png  # Confusion matrix (CNN)
│   └── confusion_matrix_svm.png  # Confusion matrix (SVM)
└── utils/
    └── helper_functions.py       # Utility functions for preprocessing
```

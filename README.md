# Readme.md
# Skin Cancer Classification Project

This Jupyter Notebook documents the skin cancer classification project, covering data acquisition, preprocessing, model training, and evaluation. The project utilizes the HAM10000 dataset (multi-class) for classifying skin lesions as cancerous or normal (Binary Classification using custom metadata.csv).

# Table of Contents
1. Mount Google Drive
2. Importing Essential Libraries
3. [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](#3-download-dataset)
4. Total Size of Files in Each Directory
5. Count Files in Each Directory
6. Reading Custom Metadata.csv
7. Upload Custom Metadata.csv
8. Checking Size of Images in Directory
9. Organizing Normal Images in Separate Directory
10. Organizing Cancer Images in Separate Directory
11. Augmenting and Visualizing Images
    - Augmenting and Visualizing a Single Image
    - Augmenting All Images in Cancer Directory
    - Counting the Number of Files in Each Directory
12. Combined Metadata
    - Combining Metadata (Normal + Cancer + Aug_Cancer)
    - Reading Combined Metadata
13. Feature Extraction
    - Extracting Features (8x8 Grayscale)
    - Loading Features.csv
    - Merging Metadata with Features
14. Label Encoding
15. Train-Test Split
16. Fitting Different ML Models
17. Model Evaluation and Visualization
    - Confusion Matrix Heatmap for Each Model
    - ROC Curve for Different Models
    - Precision-Recall Curve
    - Precision, Recall, F1 Score
    - Model Comparison - Accuracy Scores
# WorkFlow
## 1. Mount Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')
```

## 2. Importing Essential Libraries

```python
import os
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

## 3. Download Dataset

```python
# Install the Kaggle package using pip
!pip install kaggle

# Download the dataset
# Note: Make sure to configure your Kaggle API token
# The dataset is downloaded from Kaggle - 'kmader/skin-cancer-mnist-ham10000'
```

## 4. Total Size of Files in Each Directory

```python
# Calculate the total size of files in each directory
# Directory paths: '/content/Skin_Cancer/HAM10000_images_part_1', '/content/Skin_Cancer/HAM10000_images_part_2'
```

## 5. Count Files in Each Directory

```python
# Count the number of files in each directory
# Directory paths: '/content/Skin_Cancer/HAM10000_images_part_1', '/content/Skin_Cancer/HAM10000_images_part_2'
```

## 6. Reading Custom Metadata.csv

```python
# Load the custom metadata.csv file
metadata_path = '/content/HAM10000_metadata.csv'
custom_metadata = pd.read_csv(metadata_path)
custom_metadata
```

```python
# Count the occurrences of unique values in the 'dx' column of the custom_metadata DataFrame
custom_metadata['dx'].value_counts()
```

```python
# Rename 'dx' column to 'diagnosis'
custom_metadata.rename(columns={'dx': 'diagnosis'}, inplace=True)
custom_metadata
```

```python
# Saving updated metadata.csv
custom_metadata.to_csv("/content/HAM10000_metadata.csv")
```

## 7. Checking Size of Images in Directory

```python
# Function to get the size of an image
from PIL import Image
import os

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

# Function to check and print image sizes in a directory
def check_image_sizes(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  # Assuming images are in JPG format
            image_path = os.path.join(directory, filename)
            size = get_image_size(image_path)
            print(f"Image: {filename}, Size: {size}")

# Replace "path/to/your/directory_1" and "path/to/your/directory_2"
# with the actual paths to your image directories
image_directory_1 = "/content/Skin_Cancer/HAM10000_images_part_1"
image_directory_2 = "/content/Skin_Cancer/HAM10000_images_part_2"

# Check and print image sizes in the first directory
check_image_sizes(image_directory_1)

# Check and print image sizes in the second directory
check_image_sizes(image_directory_2)
```

## 8. Organizing Normal Images in Separate Directory

```python
# Function to organize normal images in a separate directory
def organize_normal_images(metadata_path, base_dir, target_diagnosis, image_dirs):
    # Implementation...
    # (Refer to the provided code snippet)
    print(f"Directories and images for {target_diagnosis} organized successfully.")

# Example usage:
metadata_path = "/content/HAM10000_metadata.csv"
base_directory = "/content/gdrive/MyDrive/Images"
diagnosis_to_organize = "Normal"
image_directories = ["/content/Skin_Cancer/HAM10000_images_part_1", "/content/Skin_Cancer/HAM10000_images_part_2"]

organize_normal_images(metadata_path, base_directory, diagnosis_to_organize, image_directories)
```

## 9. Organizing Cancer Images in Separate Directory

```python
# Function to organize cancer images in a separate directory
def organize_cancer_images(metadata_path, base_dir, target_diagnosis, image_dirs):
    # Implementation...
    # (Refer to the provided code snippet)
    print(f"Directories and images for {target_diagnosis} organized successfully.")

# Example usage:
metadata_path = "/content/HAM10000_metadata.csv"
base_directory = "/content/gdrive/MyDrive/Images"
diagnosis_to_organize = "Cancer"
image_directories = ["/content/Skin_Cancer/HAM10000_images_part_1", "/content/Skin_Cancer/HAM10000_images_part_2"]

organize_cancer_images(metadata_path, base_directory, diagnosis_to_organize, image_directories)
```

Certainly! Here's the continuation and completion of the code from "Organizing Normal Images" to the end, formatted in Jupyter-style Markdown for your README.md:

```markdown
## Organizing Normal Images

```python
# Function to organize normal images
def organize_normal_images(metadata_path, base_dir, target_diagnosis, image_dirs):
    normal_metadata = pd.read_csv(metadata_path)
    normal_metadata = normal_metadata[normal_metadata['diagnosis'] == target_diagnosis]

    for image_dir in image_dirs:
        for index, row in normal_metadata.iterrows():
            source_path = os.path.join(image_dir, row['image_id'] + '.jpg')
            destination_dir = os.path.join(base_dir, 'Normal')
            destination_path = os.path.join(destination_dir, row['image_id'] + '.jpg')

            shutil.copy(source_path, destination_path)

# Example usage
metadata_path = "/content/HAM10000_metadata.csv"
base_directory = "/content/gdrive/MyDrive/Images"
diagnosis_to_organize = "Normal"
image_directories = [
    "/content/Skin_Cancer/HAM10000_images_part_1",
    "/content/Skin_Cancer/HAM10000_images_part_2"
]

organize_normal_images(metadata_path, base_directory, diagnosis_to_organize, image_directories)
```

## Organizing Cancer Images

```python
# Function to organize cancer images
def organize_cancer_images(metadata_path, base_dir, target_diagnosis, image_dirs):
    cancer_metadata = pd.read_csv(metadata_path)
    cancer_metadata = cancer_metadata[cancer_metadata['diagnosis'] == target_diagnosis]

    for image_dir in image_dirs:
        for index, row in cancer_metadata.iterrows():
            source_path = os.path.join(image_dir, row['image_id'] + '.jpg')
            destination_dir = os.path.join(base_dir, 'Cancer')
            destination_path = os.path.join(destination_dir, row['image_id'] + '.jpg')

            shutil.copy(source_path, destination_path)

# Example usage
metadata_path = "/content/HAM10000_metadata.csv"
base_directory = "/content/gdrive/MyDrive/Images"
diagnosis_to_organize = "Cancer"
image_directories = [
    "/content/Skin_Cancer/HAM10000_images_part_1",
    "/content/Skin_Cancer/HAM10000_images_part_2"
]

organize_cancer_images(metadata_path, base_directory, diagnosis_to_organize, image_directories)
```

## Augmenting and Visualizing Images

### Augmenting and Visualizing a Single Image

```python
# Function to augment and visualize a single image
def augment_and_visualize_single_image(image_path, target_size=(600, 450), num_augmented_images=5):
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)

    # Use data augmentation to create augmented images
    data_augmentation = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Save augmented images to the specified directory
    save_dir = os.path.dirname(image_path)
    save_prefix = 'augmented_' + os.path.splitext(os.path.basename(image_path))[0]
    i = 0

    for batch in data_augmentation.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix=save_prefix, save_format='jpg'):
        i += 1
        if i >= num_augmented_images:
            break

# Example usage
image_dir = "/content/gdrive/MyDrive/Images/Cancer"
image_file = "ISIC_0024313.jpg"
image_path = os.path.join(image_dir, image_file)
augment_and_visualize_single_image(image_path)
```

### Augmenting All Images in Cancer Directory

```python
# Function to augment images in a directory
def augment_images(original_image_dir, augmented_image_dir, target_size=(600, 450), num_augmented_images=4):
    # Implementation here...

# Example usage
original_image_directory = "/content/gdrive/MyDrive/Images/Cancer"
augmented_image_directory = "/content/gdrive/MyDrive/Images/Aug_Cancer"
augment_images(original_image_directory, augmented_image_directory, target_size=(600, 450), num_augmented_images=4)
```

### Counting the Number of Files in Each Directory

```python
# Function to count files in a directory
def count_files_in_directory(directory_path):
    # Implementation here...

# Example usage
directory_path_part_1 = '/content/gdrive/MyDrive/Images/Cancer'
num_files_part_1 = count_files_in_directory(directory_path_part_1)
print(f'Number of files in Cancer: {num_files_part_1}')
```

## Combined Metadata

### Combining Metadata (Normal + Cancer + Aug_Cancer)

```python
# Function to create augmented metadata
def create_augmented_metadata(original_metadata_path, augmented_image_dir, output_metadata_path):
    # Implementation here...

# Example usage
original_metadata_path = "/content/HAM10000_metadata.csv"
augmented_image_directory = "/content/gdrive/MyDrive/Images/Aug_Cancer"
output_metadata_path = "/content/gdrive/MyDrive/Images/Aug_HAM10000_metadata.csv"

create_augmented_metadata(original_metadata_path, augmented_image_directory, output_metadata_path)
```

### Reading Combined Metadata

```python
# Reading combined metadata.csv
df = pd.read_csv('/content/gdrive/MyDrive/Images/Aug_HAM10000_metadata.csv')
df.drop(columns=['Unnamed: 0', 'file_path'], inplace=True)
df['diagnosis'].value_counts()
```

... (Continued in the next message)
```markdown
## Feature Extraction

### Extracting Features (8x8 Grayscale)

```python
# Function to extract features and save to CSV
def extract_features_and_save(image_dir_normal, image_dir_cancer, image_dir_aug_cancer, custom_metadata_path, features_csv_path):
    # Implementation here...

# Example usage
image_dir_normal = '/content/gdrive/MyDrive/Images/Normal'
image_dir_cancer = '/content/gdrive/MyDrive/Images/Cancer'
image_dir_aug_cancer = '/content/gdrive/MyDrive/Images/Aug_Cancer'
custom_metadata_path = '/content/gdrive/MyDrive/Images/Aug_HAM10000_metadata.csv'
features_csv_path = '/content/gdrive/MyDrive/Images/Aug_HAM_8_8_custom_features_normalized.csv'

extract_features_and_save(image_dir_normal, image_dir_cancer, image_dir_aug_cancer, custom_metadata_path, features_csv_path)
```

### Loading Features.csv

```python
# Loading features CSV file
features_path = '/content/gdrive/MyDrive/Images/Aug_HAM_8_8_custom_features_normalized.csv'
features_df = pd.read_csv(features_path)
features_df
```

### Merging Metadata with Features

```python
# Merging metadata with features
merged_data = pd.merge(df, features_df, on=['lesion_id', 'image_id'])
merged_data.columns
merged_data = merged_data.drop(['age', 'sex', 'localization'], axis

=1)
merged_data['diagnosis'].dtype
merged_data['diagnosis'].value_counts()
```

## Label Encoding

```python
# Label Encoding
label_encoder = LabelEncoder()
merged_data['diagnosis_label'] = label_encoder.fit_transform(merged_data['diagnosis'])
merged_data['diagnosis_label'] = 1 - merged_data['diagnosis_label']
merged_data[['diagnosis', 'diagnosis_label']]
```

## Train-Test Split

```python
# Train-test split
shuffled_data = merged_data.sample(frac=1, random_state=42)
X = shuffled_data.drop(['lesion_id', 'image_id', 'diagnosis', 'diagnosis_label'], axis=1)
y = shuffled_data['diagnosis_label']

# Checking data normalization
out_of_range_values = X[(X < 0) | (X > 1)].stack()

if not out_of_range_values.empty:
    print("Out-of-range values found:")
    print(out_of_range_values)
else:
    print("No values outside the [0, 1] range. Data is already normalized")
```

## Fitting Different ML Models

```python
# Fitting different ML models
nb_model = GaussianNB()
svm_model = SVC()
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()

nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Making predictions
nb_predictions = nb_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
```

## Model Evaluation and Visualization

### Confusion Matrix Heatmap for Each Model

```python
# Confusion matrix heatmap
models = {'Naive Bayes': nb_predictions, 'SVM': svm_predictions, 'Decision Tree': dt_predictions, 'Random Forest': rf_predictions}

for model_name, predictions in models.items():
    accuracy = accuracy_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)

    print(f"\n\nModel: {model_name}")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)

    # Plotting the heatmap
    plt.figure(figsize=(5, 3))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Cancer'], yticklabels=['Normal', 'Cancer'])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
```

### ROC Curve for Different Models

```python
# ROC Curve for different models
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_predictions)
roc_auc_nb = auc(fpr_nb, tpr_nb)

fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_predictions)
roc_auc_svm = auc(fpr_svm, tpr_svm)

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_predictions)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_predictions)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plotting ROC curves
plt.figure(figsize=(6, 4))
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label='Naive Bayes (AUC = %0.2f)' % roc_auc_nb)
plt.plot(fpr_svm, tpr_svm, color='green', lw=2, label='SVM (AUC = %0.2f)' % roc_auc_svm)
plt.plot(fpr_dt, tpr_dt, color='red', lw=2, label='Decision Tree (AUC = %0.2f)' % roc_auc_dt)
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Different Models')
plt.legend(loc="lower right")
plt.savefig('ROC Curve for Different Models.png')
plt.show()
```

### Precision-Recall Curve

```python
# Precision-Recall curve
plt.figure(figsize=(6, 4))

for model_name, predictions in models.items():
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    auc_score = auc(recall, precision)

    plt.plot(recall, precision, label=f'{model_name} (AUC = {auc_score:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('Plot Precision-Recall Curve.png')
plt.show()
```

### Precision, Recall, F1 Score

```python
# Precision, Recall, F1 Score
precision_scores = []
recall_scores = []
f1_scores = []

model_names = list(models.keys())

for model_name, predictions in models.items():
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Plot bar charts
fig, ax = plt.subplots(3, 1, figsize=(6, 4))

# Precision
ax[0].bar(model_names, precision_scores, color='blue', alpha=0.7)
ax[0].set_ylabel('Precision')
ax[0].set_title('Precision Scores')

# Recall
ax[1].bar(model_names, recall_scores, color='green', alpha=0.7)
ax[1].set_ylabel('Recall')
ax[1].set_title('Recall Scores')

# F1-score
ax[2].bar(model_names, f1_scores, color='orange', alpha=0.7)
ax[2].set_ylabel('F1-score')
ax[2].set_title('F1-score Scores')

plt.tight_layout()
plt.savefig('precision_recall_fscore_support.png')
plt.show()
```

### Model Comparison - Accuracy Scores

```python
# Model Comparison - Accuracy Scores
accuracy_nb = accuracy_score(y_test, nb_predictions)
accuracy_svm = accuracy_score(y_test, svm_predictions)
accuracy_dt = accuracy_score(y_test, dt_predictions)
accuracy_rf = accuracy_score(y_test, rf_predictions)

model_names = ['Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
accuracy_scores = [accuracy_nb, accuracy_svm, accuracy_dt, accuracy_rf]

plt.figure(figsize=(8, 6))
for i, acc in enumerate(accuracy_scores):
    plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

plt.bar(model_names, accuracy_scores, color=['blue', 'green', 'red', 'purple'])
plt.title('Model Comparison - Accuracy Scores')
plt.ylabel('Accuracy')
plt.savefig('Model Comparison - Accuracy Scores.png')
plt.show()

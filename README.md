# Readme.md
# Skin Cancer Classification Project

This Jupyter Notebook documents the skin cancer classification project, covering data acquisition, preprocessing, model training, and evaluation. The project utilizes the HAM10000 dataset (multi-class) for classifying skin lesions as cancerous or normal (Binary Classification using custom metadata.csv). In this project, the dataset comprises multi-class images but I have changed into Binary-class images through custom metadata.csv and also used GAN to overcome the imbalance of classes. The processed images are available in the following Google Drive link: [Images](https://drive.google.com/drive/folders/1--kzEJA_BiprKeUjdbNrffWoo6-WmzF-?usp=sharing). If you want to save time and do not want to make (normal, cancer, aug_cancer) folders from scratch, You can access the images folder by opening the link and adding it as a shortcut into your drive to perform various tasks such as data preprocessing, model training, and evaluation as outlined in the Jupyter Notebook to save your time and.

# Table of Contents
1. Mount Google Drive
2. Importing Essential Libraries
3. [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](#3-download-dataset)
4. Total Size of Files in Each Directory
5. Count Files in Each Directory
6. Reading Custom Metadata.csv
7. Upload Custom Metadata.csv
8. Checking the Size of Images in the Directory
9. Organizing Normal Images in a Separate Directory
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
# Kaggle Integration in Google Colab

This guide provides step-by-step instructions on how to set up Kaggle integration in Google Colab. This integration allows you to effortlessly download Kaggle datasets and participate in Kaggle competitions directly from your Colab notebooks.

## 1. Download `kaggle.json`:

- Go to the Kaggle website: [https://www.kaggle.com/](https://www.kaggle.com/)
- Log in or create a Kaggle account if you don't have one.
- Navigate to your account settings page.
- Scroll down to the "API" section.
- Click on "Create New API Token" to download the `kaggle.json` file.

## 2. Upload `kaggle.json` to Google Colab:

- Open your Google Colab notebook.
- Click on the "Files" tab in the left sidebar.
- Click on the "Upload" button.
- Select the `kaggle.json` file you downloaded and upload it to Colab.

## 3. Install Kaggle API in Colab:

Run the following commands in a Colab cell to install the Kaggle API and move the `kaggle.json` file to the appropriate directory:

```python
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
4. Use Kaggle API:
Now, you can use the Kaggle API in your Colab notebook to download datasets or interact with Kaggle competitions. For example:
```
# List available datasets
!kaggle datasets list

# Download a dataset
!kaggle datasets download -d dataset-name
```
Make sure to replace the dataset name with the actual name of the Kaggle dataset you want to download.

## 3. Download Dataset

```python
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from google.colab import files

# Dataset name
dataset_name = 'kmader/skin-cancer-mnist-ham10000'

# Directory to download the dataset
download_dir = '/content/Skin_Cancer/'  # Use the correct directory path in Colab

# Check if the dataset directory already exists
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Check if the dataset files already exist
files_in_download_dir = os.listdir(download_dir)

# Check if dataset files are already downloaded
if not any(dataset_name in file for file in files_in_download_dir):
    # Upload the Kaggle API token file
    uploaded = files.upload()

    # Move the uploaded file to the correct location
    for fn in uploaded.keys():
        os.rename(fn, '/root/.kaggle/kaggle.json')

    # Set permissions for the Kaggle API token file
    os.chmod('/root/.kaggle/kaggle.json', 600)

    # Create Kaggle API instance
    api = KaggleApi()
    api.authenticate()  # Make sure you to configure your Kaggle API token

    # Download the dataset
    api.dataset_download_files(dataset_name, path=download_dir, unzip=True)

    print("Dataset downloaded successfully!")
else:
    print("Dataset already exists. Skipping download.")

# List files in the downloaded directory
files_list = os.listdir(download_dir)
print("\nFiles in the downloaded directory:")
print(files_list)

```

## 4. Total Size of Files in Each Directory

```python
import os

# Define the directory paths
directory_path_part_1 = '/content/Skin_Cancer/HAM10000_images_part_1'
directory_path_part_2 = '/content/Skin_Cancer/HAM10000_images_part_2'

# Function to get the total size of files in a directory
def get_directory_size(directory_path):
    try:
        # Get the list of all files in the directory
        files = os.listdir(directory_path)

        # Calculate the total size of files in bytes
        total_size_bytes = sum(os.path.getsize(os.path.join(directory_path, file)) for file in files)

        # Convert bytes to gigabytes
        total_size_GB = total_size_bytes / (1024**3)

        return total_size_GB
    except FileNotFoundError:
        return 0  # Return 0 if the directory is not found

# Get total size of files in each directory
size_part_1_GB = get_directory_size(directory_path_part_1)
size_part_2_GB = get_directory_size(directory_path_part_2)

# Print the results
print(f'Total size of files in HAM10000_images_part_1: {size_part_1_GB:.2f} GB')
print(f'Total size of files in HAM10000_images_part_2: {size_part_2_GB:.2f} GB')
```

## 5. Count Files in Each Directory

```python
import os

# Define the directory path
directory_path_part_1 = '/content/Skin_Cancer/HAM10000_images_part_1'
directory_path_part_2 = '/content/Skin_Cancer/HAM10000_images_part_2'

# Function to count files in a directory
def count_files_in_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Count the number of files
        num_files = len(files)

        return num_files
    except FileNotFoundError:
        return 0  # Return 0 if the directory is not found

# Count files in each directory
num_files_part_1 = count_files_in_directory(directory_path_part_1)
num_files_part_2 = count_files_in_directory(directory_path_part_2)

# Print the results
print(f'Number of files in HAM10000_images_part_1: {num_files_part_1}')
print(f'Number of files in HAM10000_images_part_2: {num_files_part_2}')

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
def organize_normal_images(metadata_path, base_dir, target_diagnosis, image_dirs):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(metadata_path)

    # Check if the target diagnosis directory already exists
    label_dir = os.path.join(base_dir, str(target_diagnosis))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        print(f"Directory for {target_diagnosis} created.")
    else:
        print(f"Directory for {target_diagnosis} already exists.")

    # Iterate through rows and organize images only for the target diagnosis
    for index, row in df.iterrows():
        lesion_id = row['lesion_id']
        image_id = row['image_id']
        diagnosis = row['diagnosis']

        # Check if the current image has the target diagnosis
        if diagnosis == target_diagnosis:
            image_source_path = None

            # Check each image directory for the image
            for image_dir in image_dirs:
                current_image_path = os.path.join(image_dir, f"{image_id}.jpg")

                # Check if the image exists in the current directory
                if os.path.exists(current_image_path):
                    image_source_path = current_image_path
                    break

            # Check if the image exists in any of the source directories
            if image_source_path is not None:
                image_dest_path = os.path.join(label_dir, f"{image_id}.jpg")

                # Check if the image already exists in the destination directory
                if not os.path.exists(image_dest_path):
                    # Copy or move the image to the corresponding label directory
                    # You can use shutil.copy or shutil.move for this
                    # For simplicity, I'll use shutil.copy as an example
                    shutil.copy(image_source_path, image_dest_path)
                    print(f"Image {image_id} copied to {target_diagnosis} directory.")
                else:
                    print(f"Image {image_id} already exists in {target_diagnosis} directory.")
            else:
                print(f"Image {image_id} not found in any source directory.")

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
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(metadata_path)

    # Check if the target diagnosis directory already exists
    label_dir = os.path.join(base_dir, str(target_diagnosis))
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
        print(f"Directory for {target_diagnosis} created.")
    else:
        print(f"Directory for {target_diagnosis} already exists.")

    # Iterate through rows and organize images only for the target diagnosis
    for index, row in df.iterrows():
        lesion_id = row['lesion_id']
        image_id = row['image_id']
        diagnosis = row['diagnosis']

        # Check if the current image has the target diagnosis
        if diagnosis == target_diagnosis:
            image_source_path = None

            # Check each image directory for the image
            for image_dir in image_dirs:
                current_image_path = os.path.join(image_dir, f"{image_id}.jpg")

                # Check if the image exists in the current directory
                if os.path.exists(current_image_path):
                    image_source_path = current_image_path
                    break

            # Check if the image exists in any of the source directories
            if image_source_path is not None:
                image_dest_path = os.path.join(label_dir, f"{image_id}.jpg")

                # Check if the image already exists in the destination directory
                if not os.path.exists(image_dest_path):
                    # Copy or move the image to the corresponding label directory
                    # You can use shutil.copy or shutil.move for this
                    # For simplicity, I'll use shutil.copy as an example
                    shutil.copy(image_source_path, image_dest_path)
                    print(f"Image {image_id} copied to {target_diagnosis} directory.")
                else:
                    print(f"Image {image_id} already exists in {target_diagnosis} directory.")
            else:
                print(f"Image {image_id} not found in any source directory.")

    print(f"Directories and images for {target_diagnosis} organized successfully.")

# Example usage:
metadata_path = "/content/HAM10000_metadata.csv"
base_directory = "/content/gdrive/MyDrive/Images"
diagnosis_to_organize = "Cancer"
image_directories = ["/content/Skin_Cancer/HAM10000_images_part_1", "/content/Skin_Cancer/HAM10000_images_part_2"]

organize_cancer_images(metadata_path, base_directory, diagnosis_to_organize, image_directories)
```

## Augmenting and Visualizing Images

### 10. Augmenting and Visualizing a Single Image

```python
# Function to augment and visualize a single image
def augment_and_visualize_single_image(image_path, target_size=(600, 450), num_augmented_images=5):
    # Load the single image
    img = image.load_img(image_path, target_size=target_size)

    # Convert the image to a numpy array
    x = image.img_to_array(img)

    # Reshape the image to (1, height, width, channels) as required by the flow() method
    x = np.expand_dims(x, axis=0)

    # Create an instance of the ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generate augmented images and visualize them
    plt.figure(figsize=(10, 15))

    # Display the original image
    plt.subplot(num_augmented_images + 1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # Display augmented images
    for i, batch in enumerate(datagen.flow(x, batch_size=1), start=2):
        plt.subplot(num_augmented_images + 1, 3, i)
        plt.imshow(image.array_to_img(batch[0]))
        plt.title(f'Augmented Image {i-1}')
        plt.axis('off')
        if i > (num_augmented_images):  # Display the specified number of augmented images
            break

    plt.tight_layout()
    plt.show()

# Path to the directory containing the single image
image_dir = "/content/gdrive/MyDrive/Images/Cancer"

# Specify the image file name
image_file = "ISIC_0024313.jpg"

# Full path to the image
image_path = os.path.join(image_dir, image_file)

# Call the function to augment and visualize a single image
augment_and_visualize_single_image(image_path)
```

### 11. Augmenting All Images in Cancer Directory

```python
# Function to augment images in a directory
def augment_images(original_image_dir, augmented_image_dir, target_size=(600, 450), num_augmented_images=4):
    """
    Augment images in the original directory and save the augmented images in the output directory.

    Parameters:
    - original_image_dir (str): Path to the directory containing original images.
    - augmented_image_dir (str): Path to the directory where augmented images will be saved.
    - target_size (tuple): Target size for the images (height, width).
    - num_augmented_images (int): Number of augmented images to generate per original image.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(augmented_image_dir):
        os.makedirs(augmented_image_dir)

    # Create an instance of the ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Get the list of image files in the original directory
    image_files = [f for f in os.listdir(original_image_dir) if f.endswith('.jpg')]

    # Iterate through each image file and apply augmentation
    for img_file in image_files:
        # Get the image ID from the file name
        image_id = os.path.splitext(img_file)[0]

        print(f"Augmenting images for: {image_id}")

        # Load the image
        img_path = os.path.join(original_image_dir, img_file)
        img = image.load_img(img_path, target_size=target_size)

        # Convert the image to a numpy array
        x = image.img_to_array(img)

        # Reshape the image to (1, height, width, channels) as required by the flow() method
        x = np.expand_dims(x, axis=0)

        # Generate augmented images and save them to the output directory
        for i in range(1, num_augmented_images + 1):
            batch = next(datagen.flow(x, batch_size=1))
            augmented_image_path = os.path.join(augmented_image_dir, f'{image_id}_aug_{i}.jpg')
            image.save_img(augmented_image_path, batch[0])
            print(f"Saved augmented image {i}/{num_augmented_images}: {augmented_image_path}")

        print(f"Augmentation complete for: {image_id}")
        print(f"Original Image: {image_id}.jpg, Augmented Images: {num_augmented_images}")

# Example usage:
original_image_directory = "/content/gdrive/MyDrive/Images/Cancer"
augmented_image_directory = "/content/gdrive/MyDrive/Images/Aug_Cancer"
augment_images(original_image_directory, augmented_image_directory, target_size=(600, 450), num_augmented_images=4)
```

### 12. Counting the Number of Files in Cancer Directory

```python
# Function to count files in a Cancer directory
import os

# Define the Cancer directory path
directory_path_part_1 = '/content/gdrive/MyDrive/Images/Cancer'

# Function to count files in a directory
def count_files_in_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Count the number of files
        num_files = len(files)

        return num_files
    except FileNotFoundError:
        return 0  # Return 0 if the directory is not found

# Count files in each directory
num_files_part_1 = count_files_in_directory(directory_path_part_1)

# Print the results
print(f'Number of files in Cancer: {num_files_part_1}')
```
### 13. Counting the Number of Files in Aug_Cancer Directory

```python
# Function to count files in a Aug_Cancer directory
import os

# Define the Aug_Cancer directory path
directory_path_part_1 = '/content/gdrive/MyDrive/Images/Aug_Cancer'

# Function to count files in a directory
def count_files_in_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Count the number of files
        num_files = len(files)

        return num_files
    except FileNotFoundError:
        return 0  # Return 0 if the directory is not found

# Count files in each directory
num_files_part_1 = count_files_in_directory(directory_path_part_1)

# Print the results
print(f'Number of files in Aug_Cancer: {num_files_part_1}')
```

### 14. Counting the Number of Files in Normal Directory

```python
# Function to count files in a Normal directory
import os

# Define the directory path
directory_path_part_1 = '/content/gdrive/MyDrive/Images/Normal'

# Function to count files in a directory
def count_files_in_directory(directory_path):
    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Count the number of files
        num_files = len(files)

        return num_files
    except FileNotFoundError:
        return 0  # Return 0 if the directory is not found

# Count files in each directory
num_files_part_1 = count_files_in_directory(directory_path_part_1)

# Print the results
print(f'Number of files in Normal: {num_files_part_1}')
```

## Combined Metadata

### 15. Combining Metadata (Normal + Cancer + Aug_Cancer)

```python
# Function to create augmented metadata
import os
import pandas as pd

def create_augmented_metadata(original_metadata_path, augmented_image_dir, output_metadata_path):
    # Load the original metadata
    original_metadata = pd.read_csv(original_metadata_path)

    # Get the list of image files in the augmented directory
    augmented_image_files = [f for f in os.listdir(augmented_image_dir) if f.endswith('.jpg')]

    # Create a new DataFrame to store augmented metadata
    augmented_metadata = pd.DataFrame(columns=original_metadata.columns)

    for augmented_image_file in augmented_image_files:
        # Extract information from the augmented image filename
        file_parts = os.path.splitext(augmented_image_file)[0].split('_')
        original_image_id = '_'.join(file_parts[:-2])  # Extract the original image ID
        augmentation_number = int(file_parts[-1])  # Extract the augmentation number

        # Find the corresponding row in the original metadata
        original_row = original_metadata[original_metadata['image_id'] == original_image_id]

        # Check if the original image is present in the original metadata
        if not original_row.empty:
            # Create a new row for the augmented image
            augmented_row = original_row.copy()
            augmented_row['image_id'] = f"{original_image_id}_aug_{augmentation_number}"
            augmented_row['file_path'] = os.path.join(augmented_image_dir, augmented_image_file)

            # Append the augmented row to the new metadata
            augmented_metadata = pd.concat([augmented_metadata, augmented_row], ignore_index=True)

    # Concatenate the original and augmented metadata
    combined_metadata = pd.concat([original_metadata, augmented_metadata], ignore_index=True)

    # Save the combined metadata to a new CSV file
    combined_metadata.to_csv(output_metadata_path, index=False)
    print(f"Combined metadata saved to {output_metadata_path}")

# Example usage:
original_metadata_path = "/content/HAM10000_metadata.csv"
augmented_image_directory = "/content/gdrive/MyDrive/Images/Aug_Cancer"
output_metadata_path = "/content/gdrive/MyDrive/Images/Aug_HAM10000_metadata.csv"

create_augmented_metadata(original_metadata_path, augmented_image_directory, output_metadata_path)
```

### 16. Reading Combined Metadata
```
df = pd.read_csv('/content/gdrive/MyDrive/Images/Aug_HAM10000_metadata.csv')
df
```
```
# dropping "unnamed: 0" column from dataframe
df.drop(columns=['Unnamed: 0', 'file_path'], inplace=True)
df
```
```
# Count the occurrences of each unique value in the 'diagnosis' column of the DataFrame
df['diagnosis'].value_counts()
```

## Feature Extraction

### 17. Extracting Features (8x8 Grayscale)

```python
# Function to extract features and save to CSV
from skimage import io, color, transform

def extract_features_and_save(image_dir_normal, image_dir_cancer, image_dir_aug_cancer, custom_metadata_path, features_csv_path):
    """
    Extract features from images in three directories (Normal, Cancer, Aug_Cancer) based on custom metadata and save to CSV.

    Parameters:
    - image_dir_normal (str): Path to the directory containing Normal images.
    - image_dir_cancer (str): Path to the directory containing Cancer images.
    - image_dir_aug_cancer (str): Path to the directory containing Aug_Cancer images.
    - custom_metadata_path (str): Path to the custom metadata CSV file.
    - features_csv_path (str): Path to save the extracted features CSV file.

    Returns:
    - None
    """
    # Load custom metadata
    custom_metadata = pd.read_csv(custom_metadata_path)

    # Initialize an empty DataFrame to store features
    features_df = pd.DataFrame()

    # Check if the features CSV file already exists
    if os.path.exists(features_csv_path):
        print("Features CSV file already exists. Skipping feature extraction.")
        return
    else:
        # Loop through each row in the custom metadata
        for index, row in custom_metadata.iterrows():
            lesion_id = row['lesion_id']
            image_id = row['image_id']

            # Construct the full path to the image file
            if os.path.exists(os.path.join(image_dir_normal, f"{image_id}.jpg")):
                image_path = os.path.join(image_dir_normal, f"{image_id}.jpg")
            elif os.path.exists(os.path.join(image_dir_cancer, f"{image_id}.jpg")):
                image_path = os.path.join(image_dir_cancer, f"{image_id}.jpg")
            elif os.path.exists(os.path.join(image_dir_aug_cancer, f"{image_id}.jpg")):
                image_path = os.path.join(image_dir_aug_cancer, f"{image_id}.jpg")
            else:
                print(f"Image not found for {image_id}. Skipping.")
                continue

            # Read and resize the image to 8x8 pixels in grayscale
            image = io.imread(image_path)
            image_gray = color.rgb2gray(image)
            image_resized = transform.resize(image_gray, (8, 8), anti_aliasing=True)

            # Flatten the pixel values and create a DataFrame row
            features_row = pd.DataFrame(image_resized.flatten()).transpose()

            # Normalize pixel values to the range [0, 1]
            normalized_features = (features_row - features_row.min().min()) / (features_row.max().max() - features_row.min().min())

            # Include 'image_id' and 'lesion_id' in the features DataFrame
            normalized_features['image_id'] = image_id
            normalized_features['lesion_id'] = lesion_id

            # Append the row to the features DataFrame
            features_df = pd.concat([features_df, normalized_features], ignore_index=True)

        # Save the features DataFrame to a CSV file
        features_df.to_csv(features_csv_path, index=False)
        print(f"Features saved to {features_csv_path}")

# Example usage:
image_dir_normal = '/content/gdrive/MyDrive/Images/Normal'
image_dir_cancer = '/content/gdrive/MyDrive/Images/Cancer'
image_dir_aug_cancer = '/content/gdrive/MyDrive/Images/Aug_Cancer'
custom_metadata_path = '/content/gdrive/MyDrive/Images/Aug_HAM10000_metadata.csv'
features_csv_path = '/content/gdrive/MyDrive/Images/Aug_HAM_8_8_custom_features_normalized.csv'

extract_features_and_save(image_dir_normal, image_dir_cancer, image_dir_aug_cancer, custom_metadata_path, features_csv_path)
```

### 18. Loading Features.csv

```python
# Loading features CSV file
features_path = '/content/gdrive/MyDrive/Images/Aug_HAM_8_8_custom_features_normalized.csv'
features_df = pd.read_csv(features_path)
features_df
```

### 19. Merging Metadata with Features

```python
# Merging metadata with features
merged_data = pd.merge(df, features_df, on=['lesion_id', 'image_id'])
merged_data.columns
```
```
merged_data = merged_data.drop(['age', 'sex', 'localization'], axis =1)
merged_data['diagnosis'].dtype
merged_data['diagnosis'].value_counts()
```
```
# Convert the column to string type
merged_data['diagnosis'] = merged_data['diagnosis'].astype(str)
```

## 20. Label Encoding

```python
# Label Encoding
label_encoder = LabelEncoder()
merged_data['diagnosis_label'] = label_encoder.fit_transform(merged_data['diagnosis'])
merged_data['diagnosis_label'] = 1 - merged_data['diagnosis_label']
merged_data[['diagnosis', 'diagnosis_label']]
```

## 21. Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Shuffle the data
shuffled_data = merged_data.sample(frac=1, random_state=42)

# Split the shuffled data into features (X) and labels (y)
X = shuffled_data.drop(['lesion_id', 'image_id', 'diagnosis', 'diagnosis_label'], axis=1)
y = shuffled_data['diagnosis_label']

# Split the shuffled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## 22. Checking Normalization Status
```
# Check if any value is outside the [0, 1] range
out_of_range_values = X[(X < 0) | (X > 1)].stack()

# If there are any out-of-range values, display them
if not out_of_range_values.empty:
    print("Out-of-range values found:")
    print(out_of_range_values)
else:
    print("No values outside the [0, 1] range. Data is already normalized")
```

## 23. Fitting Different ML Models

```python
# Fitting different ML models
# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
```

## Model Evaluation and Visualization

### 24. Confusion Matrix Heatmap for Each Model

```python
# Create a heatmap for each model
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

### 25. ROC Curve for Different Models

```python
# ROC Curve for different models
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have predictions for each model: nb_predictions, svm_predictions, dt_predictions, rf_predictions
# Replace them with the actual predictions from your models

# Compute ROC curve and ROC area for each class
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_predictions)
roc_auc_nb = auc(fpr_nb, tpr_nb)

fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_predictions)
roc_auc_svm = auc(fpr_svm, tpr_svm)

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_predictions)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_predictions)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curves
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
### 26. ROC-AUC score
```
from sklearn.metrics import roc_auc_score

# Print ROC AUC Score
for model_name, predictions in models.items():
    auc_score = roc_auc_score(y_test, predictions)
    print(f"The ROC AUC score for {model_name} is: {auc_score:.2f}")
```
### 27. Precision-Recall Curve

```python
# Precision-Recall curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Plot Precision-Recall Curve
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

### 28. Precision, Recall, F1 Score

```python
# Precision, Recall, F1 Score
from sklearn.metrics import precision_recall_fscore_support

# Initialize lists to store metric scores
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

### 29. Model Comparison - Accuracy Scores

```python
# Calculate accuracy scores for each model
accuracy_nb = accuracy_score(y_test, nb_predictions)
accuracy_svm = accuracy_score(y_test, svm_predictions)
accuracy_dt = accuracy_score(y_test, dt_predictions)
accuracy_rf = accuracy_score(y_test, rf_predictions)

# Plot a bar chart with exact accuracy values using matplotlib
model_names = ['Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest']
accuracy_scores = [accuracy_nb, accuracy_svm, accuracy_dt, accuracy_rf]

plt.figure(figsize=(8, 6))
for i, acc in enumerate(accuracy_scores):
    plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')

plt.bar(model_names, accuracy_scores, color=['darkorange', 'green', 'red', 'blue'])
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison - Accuracy Scores')
plt.savefig('Model Comparison - Accuracy Scores.png')
plt.show()
```
### 30. Model's Accuracy in Plotly Library
```
import plotly.io as pio
import plotly.express as px

# Calculate accuracy scores for each model
accuracy_nb = accuracy_score(y_test, nb_predictions)
accuracy_svm = accuracy_score(y_test, svm_predictions)
accuracy_dt = accuracy_score(y_test, dt_predictions)
accuracy_rf = accuracy_score(y_test, rf_predictions)

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Models': ['Naive Bayes', 'SVM', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_nb, accuracy_svm, accuracy_dt, accuracy_rf]
})

# Plot with plotly express
fig = px.bar(df, x='Models', y='Accuracy', text='Accuracy', color='Models',
             labels={'Accuracy': 'Accuracy'},
             title='Model Comparison - Accuracy Scores',
             hover_data=['Accuracy'],
             template='plotly_white',
             color_discrete_map={'Naive Bayes': 'purple', 'SVM': 'green', 'Decision Tree': 'red', 'Random Forest': 'blue'},
             width=1000, height=700)

# Update layout to add hover effects
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

# Show the plot
fig.show()
```
# 31. Results
## GAN for Class Imbalance: Results and Analysis

In an attempt to address class imbalance, a Generative Adversarial Network (GAN) was applied to augment the dataset. The process involved generating 4 augmented images for every original cancer image. After applying GAN on 1727 images of cancer, a total of 8135 images was obtained (1727 original + 6508 augmented) for the cancer class.

## Impact on Model Performance

Despite the significant increase in the number of cancer images, the overall impact on model accuracy was not substantial. In fact, there was a slight decrease in accuracy.

## Performance Evaluation: ROC AUC Scores

The ROC AUC (Receiver Operating Characteristic Area Under the Curve) score is a vital metric that quantifies the area under the ROC curve. A higher ROC AUC score generally indicates better model performance.

- AUC scores closer to 1.0 signify better model performance.
- AUC scores around 0.5 suggest that the model's performance is not significantly better than random chance.

## Performance vs ROC Curve: What is Preferable and Why?

Considering both the ROC AUC scores and the top-left positioning on the ROC curve:

- **SVM and Random Forest**: Show relatively good performance.
- **Naive Bayes and Decision Tree**: Exhibit moderate performance.

### Why ROC AUC Score?

The ROC AUC score is preferable as it provides a comprehensive measure of a model's ability to distinguish between classes. Key points to consider:

- **Closer to 1.0**: Indicates better model performance.
- **Around 0.5**: Suggests that the model's performance is not significantly better than random chance.

# 32. Conclusion

The GAN-based augmentation strategy did not result in a substantial increase in accuracy. The ROC AUC scores provide insights into the relative performance of different models, with SVM and Random Forest exhibiting relatively better performance.
## ROC-Curve without GAN
<img src="https://github.com/selectTopic/Selected-Topics/assets/153743838/69691692-c6c1-4f95-abc4-24c7f34a00d6" width="500">

## ROC-Curve with GAN
<img src="https://github.com/selectTopic/Selected-Topics/assets/153743838/c77f1047-416f-4e6a-a687-c8e56d4732ef" width="500">

## Model-Performance without GAN
<img src="https://github.com/selectTopic/Selected-Topics/assets/153743838/8b47d87d-3d5b-4b7f-852a-104a2d50cd49" width="600">

## Model-Performance with GAN
<img src="https://github.com/selectTopic/Selected-Topics/assets/153743838/f34dffc6-7d93-417e-8019-1f7fe4e7f50f" width="600">

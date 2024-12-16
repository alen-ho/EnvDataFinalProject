
# --------------------------------------------------------------------------------------------------------------------
# My goal is to identify any Construction and Demolition (C&D) waste landfills within the area of interest.
# I have downloaded the .tif image data from Esri World Imagery for this purpose.
# https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9
# --------------------------------------------------------------------------------------------------------------------

import os
import random
from osgeo import gdal
import numpy as np

# Function to read a TIFF file
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + " file cannot be opened")
    return dataset


# Function to save a TIFF file
# Output file after clipping
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # Create file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # Write affine transformation parameters
        dataset.SetProjection(im_proj)  # Write projection information
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


'''
Sliding window clipping function
TifPath: Path to the input image
SavePath: Directory to save the clipped output
CropSize: Clipping size (height and width of the clip)
RepetitionRate: Overlap rate between adjacent clips
'''
def TifCrop(TifPath, SavePath, CropSize, RepetitionRate):
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # Get image data

    # Get the current number of files in the directory and use it as the starting file name for new clips
    new_name = len(os.listdir(SavePath)) + 1

    # Clip the image using the specified overlap rate (RepetitionRate)
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            # If the image is single-band
            if len(img.shape) == 2:
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            # If the image is multi-band
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            # Write the clipped image to a file
            writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
            # Increment the file name
            new_name += 1

    # Clip the last column
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if len(img.shape) == 2:
            cropped = img[int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        else:
            cropped = img[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        # Write the clipped image to a file
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name += 1

    # Clip the last row
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if len(img.shape) == 2:
            cropped = img[(height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        # Write the clipped image to a file
        writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
        new_name += 1

    # Clip the bottom-right corner
    if len(img.shape) == 2:
        cropped = img[(height - CropSize): height,
                  (width - CropSize): width]
    else:
        cropped = img[:,
                  (height - CropSize): height,
                  (width - CropSize): width]
    # Write the clipped image to a file
    writeTiff(cropped, geotrans, proj, SavePath + "/%d.tif" % new_name)
    new_name += 1


'''
Random cropping function
ImagePath: Path to the original image
LabelPath: Path to the label image
IamgeSavePath: Directory to save cropped original images
LabelSavePath: Directory to save cropped label images
CropSize: Size of the crop (height and width)
CutNum: Number of crops to generate
'''

def RandomCrop(ImagePath, LabelPath, IamgeSavePath, LabelSavePath, CropSize, CutNum):
    dataset_img = readTif(ImagePath)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)  # Get original image data
    dataset_label = readTif(LabelPath)
    label = dataset_label.ReadAsArray(0, 0, width, height)  # Get label data

    # Get the current number of files in the directory and use it as the starting file name for new crops
    fileNum = len(os.listdir(IamgeSavePath))
    new_name = fileNum + 1
    while new_name < CutNum + fileNum + 1:
        # Generate the top-left XY coordinates for the crop
        UpperLeftX = random.randint(0, height - CropSize)
        UpperLeftY = random.randint(0, width - CropSize)
        if len(img.shape) == 2:
            imgCrop = img[UpperLeftX: UpperLeftX + CropSize,
                          UpperLeftY: UpperLeftY + CropSize]
        else:
            imgCrop = img[:,
                          UpperLeftX: UpperLeftX + CropSize,
                          UpperLeftY: UpperLeftY + CropSize]
        if len(label.shape) == 2:
            labelCrop = label[UpperLeftX: UpperLeftX + CropSize,
                              UpperLeftY: UpperLeftY + CropSize]
        else:
            labelCrop = label[:,
                              UpperLeftX: UpperLeftX + CropSize,
                              UpperLeftY: UpperLeftY + CropSize]
        # Save the cropped original image and label image
        writeTiff(imgCrop, geotrans, proj, IamgeSavePath + "/%d.tif" % new_name)
        writeTiff(labelCrop, geotrans, proj, LabelSavePath + "/%d.tif" % new_name)
        new_name += 1

# Training set Generation
# Generate 300 256×256 training dataset images through random cropping
RandomCrop(r"C:\Users\jojo Y\Desktop\a\test\0194.tif",
           r"C:\Users\jojo Y\Desktop\a\label\l_31.tif",
           r"C:\Users\jojo Y\Desktop\a\result_rand",
           r"C:\Users\jojo Y\Desktop\a\result_rand_1",
           512, 5)


# Test set Generation
# Crop the image into 256×256 segments with a repetition rate of 0
TifCrop(r"C:\Users\jojo Y\Desktop\a\test\0194.tif",
        r"C:\Users\jojo Y\Desktop\a\result_seq", 512, 0)
TifCrop(r"C:\Users\jojo Y\Desktop\a\label\l_31.tif",
        r"C:\Users\jojo Y\Desktop\a\result_seq_l", 512, 0)

import os
import shutil
import random


def split_dataset(image_dir, mask_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
        image_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing masks.
        output_dir (str): Path to the output dataset directory.
        train_ratio (float): Proportion of training data.
        val_ratio (float): Proportion of validation data.
        test_ratio (float): Proportion of test data.
        seed (int): Random seed for reproducibility.
    """
    # Ensure the ratios sum up to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

    # Get all file names from the image directory
    file_names = os.listdir(image_dir)

    # Shuffle file names
    random.seed(seed)
    random.shuffle(file_names)

    # Split the file names
    total_files = len(file_names)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = file_names[:train_end]
    val_files = file_names[train_end:val_end]
    test_files = file_names[val_end:]

    # Helper function to copy files
    def copy_files(file_list, split):
        for file_name in file_list:
            # Copy image
            shutil.copy(os.path.join(image_dir, file_name),
                        os.path.join(output_dir, split, 'images', file_name))
            # Copy mask (ensure corresponding mask file exists)
            mask_name = file_name  # Assuming mask file name matches image file name
            shutil.copy(os.path.join(mask_dir, mask_name),
                        os.path.join(output_dir, split, 'masks', mask_name))

    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Dataset split complete!")
    print(f"Train: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")


# Paths
image_dir = r"C:\MyNuts\image_255"  # Path to the image folder
mask_dir = r"C:\MyNuts\label_255"  # Path to the mask folder
output_dir = r"C:\MyNuts\dataset"  # Path to the output dataset folder

# Split the dataset
split_dataset(image_dir, mask_dir, output_dir)

import os
import shutil
import random


def split_dataset(image_dir, mask_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split the dataset into training, validation, and test sets.
    Parameters:
        image_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing masks.
        output_dir (str): Path to the output dataset directory.
        train_ratio (float): Proportion of training data.
        val_ratio (float): Proportion of validation data.
        test_ratio (float): Proportion of test data.
        seed (int): Random seed for reproducibility.
    """
    # Debug print to check ratios
    print(f"Train Ratio: {train_ratio}, Val Ratio: {val_ratio}, Test Ratio: {test_ratio}")
    print(f"Total: {train_ratio + val_ratio + test_ratio}")

    # Ensure the ratios sum up to 1 with tolerance for floating-point precision
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6, "Ratios must sum to 1."

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

    # Get all file names from the image directory
    file_names = os.listdir(image_dir)

    # Shuffle file names
    random.seed(seed)
    random.shuffle(file_names)

    # Split the file names
    total_files = len(file_names)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = file_names[:train_end]
    val_files = file_names[train_end:val_end]
    test_files = file_names[val_end:]

    # Helper function to copy files
    def copy_files(file_list, split):
        for file_name in file_list:
            # Copy image
            shutil.copy(os.path.join(image_dir, file_name),
                        os.path.join(output_dir, split, 'images', file_name))
            # Copy mask (ensure corresponding mask file exists)
            mask_name = file_name  # Assuming mask file name matches image file name
            shutil.copy(os.path.join(mask_dir, mask_name),
                        os.path.join(output_dir, split, 'masks', mask_name))

    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Dataset split complete!")
    print(f"Train: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")


# Paths
image_dir = r"image_255"  # Path to the image folder
mask_dir = r"label_255"  # Path to the mask folder
output_dir = r"dataset"  # Path to the output dataset folder

# Split the dataset
split_dataset(image_dir, mask_dir, output_dir)











import os
import shutil
import random


def split_dataset(image_dir, mask_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split the dataset into training, validation, and test sets.
    Parameters:
        image_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing masks.
        output_dir (str): Path to the output dataset directory.
        train_ratio (float): Proportion of training data.
        val_ratio (float): Proportion of validation data.
        test_ratio (float): Proportion of test data.
        seed (int): Random seed for reproducibility.
    """
    # Debug print to check ratios
    print(f"Train Ratio: {train_ratio}, Val Ratio: {val_ratio}, Test Ratio: {test_ratio}")
    print(f"Total: {train_ratio + val_ratio + test_ratio}")

    # Ensure the ratios sum up to 1 with tolerance for floating-point precision
    assert abs(train_ratio + val_ratio + test_ratio - 1) < 1e-6, "Ratios must sum to 1."

    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

    # Get all .jpg file names from the image directory
    file_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # Shuffle file names
    random.seed(seed)
    random.shuffle(file_names)

    # Split the file names
    total_files = len(file_names)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)

    train_files = file_names[:train_end]
    val_files = file_names[train_end:val_end]
    test_files = file_names[val_end:]

    # Helper function to copy files
    def copy_files(file_list, split):
        for file_name in file_list:
            # Copy image
            shutil.copy(os.path.join(image_dir, file_name),
                        os.path.join(output_dir, split, 'images', file_name))
            # Copy mask (replace .jpg with .png for corresponding mask)
            mask_name = file_name.replace('.jpg', '.png')
            shutil.copy(os.path.join(mask_dir, mask_name),
                        os.path.join(output_dir, split, 'masks', mask_name))

    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Dataset split complete!")
    print(f"Train: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")


# Paths
image_dir = r"image"  # Path to the image folder
mask_dir = r"mask_label"  # Path to the mask folder
output_dir = r"dataset"  # Path to the output dataset folder

# Split the dataset
split_dataset(image_dir, mask_dir, output_dir)

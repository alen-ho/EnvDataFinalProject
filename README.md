# 1. Introduction:

Globally, more than 10 billion tons of construction and demolition waste (CDW) are generated annually (Yazdani et al., 2021), with most of it being disposed of in landfills. Illegal landfills pose significant risks, including environmental degradation, economic losses, and societal harm, such as potential landslide incidents (Nikulishyn et al., 2020). Effective recognition and monitoring of illegal landfills are crucial for governments to adopt timely measures to mitigate risks and ensure public safety. However, traditional manual methods for identifying illegal landfills are labor-intensive and inadequate for large-scale detection (Ramos et al., 2022).&#x20;

As a result, innovative technologies such as remote sensing and geographic information systems (GIS) are increasingly being adopted to manage landfills (Singh, 2019). Silvestri and Omri (2008) utilized IKONOS satellite data to identify unknown landfills in parts of the Venice Lagoon basin by analyzing the spectral characteristics of vegetation affected by known illegal landfills. Similarly, Lucendo-Monedero et al. (2015) detected 120 potential illegal landfills in Andalusia using remote sensing imagery and applied logistic regression to assess the relationship between spatial and behavioral variables, predicting the likelihood of illegal landfill occurrences. Karimi et al. (2022) combined Suomi NPP and Landsat 8 satellite data to analyze nighttime lighting, the enhanced soil-adjusted vegetation index, and land surface temperature in Saskatoon, Canada. They integrated vector data, such as road length, railway length, and the number of landfills, through a weighted approach to generate a probability map of illegal landfill distribution. Zhou et al. (2021a) explored optimal machine learning methods on the Google Earth Engine platform for identifying construction and demolition waste (CDW) accumulation points, comparing the performance of classification and regression trees, random forests, and support vector machines.&#x20;

Although these studies successfully identified landfills, traditional machine learning methods heavily depend on manually selected shallow features for statistical analysis. This reliance on human-designed characteristics limits their ability to fully achieve automatic identification of CDW landfills. However, with the advancement of deep learning in computer vision, representative semantic segmentation models have demonstrated remarkable performance in image recognition tasks. These models offer significant potential for the automatic and efficient identification of CDW landfills across large areas (Wang et al 2022). Semantic segmentation models are end-to-end frameworks that can automatically establish intrinsic connections between low-level image features and high-level semantic information (Wu et al 2024). These models train black-box systems to extract deep features tailored to specific image recognition tasks, enabling direct identification results from input images. This project proposes a method for the automatic identification of CDW landfills using semantic segmentation models (Yong et al 2023). Specifically, it involves constructing a dedicated semantic segmentation dataset for CDW landfills and training the DeepLabv3+ model to perform semantic segmentation, facilitating the automatic identification of CDW landfills.

# **2. Data and Methods**

## 2.1 Project aim

To predict the locations of landfills in urban areas using remote sensing imagery, addressing the challenges of identifying and monitoring illegal landfills.

## 2.2 Model Name: DeepLabv3+

DeepLabv3+ is a deep learning model designed for semantic segmentation. It employs Atrous Spatial Pyramid Pooling (ASPP) and an encoder-decoder architecture, enabling efficient capture of contextual information while generating precise boundary segmentation (Fang 2024).

## 2.3 Software Requirements

Ensure you have the following installed in your Python environment:

- Python 3.8+

- TensorFlow 2.8+ (Tested on TensorFlow 2.9.1)

- NumPy

- Keras

- Matplotlib (optional, for visualizing training progress)

- PIL (Python Imaging Library, for image loading and preprocessing)

## 2.4 Hardware Requirements

The model is computationally intensive and benefits from GPU acceleration. The recommended hardware specifications are as follows:

- **GPU**: NVIDIA GPU with CUDA Compute Capability 5.0+ (e.g., NVIDIA GTX 1060, RTX 2060, or higher)

- **VRAM**: At least 4 GB of dedicated GPU memory (8 GB or more recommended for larger batch sizes or higher resolutions)

- **RAM**: 16 GB or more system memory

- **Processor**: Multi-core CPU (Intel i5/i7, AMD Ryzen 5/7, or equivalent)

- **Storage**: At least 20 GB free for datasets, checkpoints, and model outputs

For best performance, ensure you have the latest drivers for your GPU and install the required CUDA and cuDNN versions compatible with TensorFlow.

## 3. Dataset

## 3.1 Original Data Categories and Sources

The data for this project is categorized into two primary types: remote sensing data and landfill location data. These datasets provide comprehensive information for the semantic segmentation task of identifying landfill boundaries in remote sensing images.

## 3.1.1 Remote Sensing Data

- **Description**:\
  The remote sensing data consists of high-resolution optical imagery provided by the **Esri World Imagery** service. These images have undergone precise geometric and atmospheric corrections, ensuring they are suitable for cartography, spatial analysis, and visualization. The dataset is globally recognized for its accuracy and quality, making it ideal for semantic segmentation tasks.

- **Data Access and Version**:\
  The data is accessed via the **Web Map Tile Service (WMTS)** from Esri, which allows seamless integration with geospatial tools like ArcGIS. The version used in this project is from **Esri World Imagery (2024)**. This version incorporates the latest updates and corrections as of the year 2024.

- **Date of Acquisition**:\
  Imagery data was accessed and downloaded during **January to February 2024** using the ArcGIS platform.

## 3.1.2 Landfill Location Data

- **Description**:\
  Landfill location data comprises the geographic coordinates of landfill sites. This data is derived from publicly available datasets hosted by Chinese government websites. The latitude and longitude of each landfill site were determined using **Google Maps**.

## 3.2 Data Preprocessing Steps

### 3.2.1. Data Annotation

- **Process**:

  - Using ArcGIS, manually annotate landfill boundaries to generate binary mask files (landfill pixels are labeled as `1`, and all others as `0`).

- **Output Format**:

  - `.png` binary mask files corresponding to the remote sensing images.

### 3.2.2 Image Segmentation

- **Process**:

  - Use Python and Pillow (PIL) to split large images into smaller tiles of **512x512 pixels**.

- **Result**:

  - The segmented dataset includes a total of **2,888 images**.

### 3.2.3 Data Splitting

- **Process**:

  - Randomly divide the dataset into:

    - **Training set**: 70% (2,006 images)

    - **Validation set**: 20%

    - **Test set**: 10%

  - Save path structure:

    ```bash
    ├── dataset/
    │   ├── train/
    │   │   ├── images/    # Training images
    │   │   ├── masks/     # Corresponding masks (binary, .png format)
    │   ├── val/
    │       ├── images/    # Validation images
    │       ├── masks/     # Corresponding masks (binary, .png format)
    ```

- **Example landfill RGB image**

![](README_md_files/53a58210-bab4-11ef-bac3-29376acdd9c9_20241215151512.jpeg?v=1&type=image&token=V1%3AhkEsUBRq9N5mFn7AdXiT2g9XtDZ8GssxDz0vUd6DDiI)

- **Example landfill mask**

![](README_md_files/69c40b70-bab4-11ef-bac3-29376acdd9c9_20241215151549.jpeg?v=1&type=image&token=V1%3Adz8pDTfGzzXPjLSNLCxC3mCywE3RU9t4w4_o-DXlpWE)

# 4. Model Training

## 4.1 Predictors and Target Variables

### Predictors (Inputs)

- **Type**: RGB remote sensing images normalized to **\[0, 1]**.

- **Format**: Standardized images of size **512x512x3**, where `3` represents the number of RGB channels.

- The structure of the input data must follow the organization outlined in **Section 3.2.3**.

### Target Variables (Outputs)

- **Type**: Binary segmentation masks where each pixel is classified as landfill (`1`) or non-landfill (`0`).

- **Format**: Binary masks of the same dimensions as the input images.

## 4.2 Training and Validation Process

### Model Training Method

- **Process**:

  - Train the model using deep learning optimization algorithms, updating network weights through backpropagation to minimize the loss function.

- **Loss Function**:

  - **Cross-Entropy Loss**: Evaluates the classification error for each pixel to ensure the model accurately distinguishes between landfill and non-landfill pixels.

### Model Validation

- **Process**:

  - Use the validation set (**20% of the total dataset**) to evaluate model performance in real-time and prevent overfitting.

## 4.3 Segmentation Performance Evaluation Metrics

1.  Pixel Accuracy (Accuracy)

    - **Definition**: Measures the overall accuracy of pixel classification.

2.  Intersection over Union (IoU)

    - **Definition**: Calculates the ratio of the intersection to the union of the predicted landfill area and the ground truth, reflecting the spatial overlap of the segmentation results.

# 5. Result and Discussion

## 5.1 Overall Pixel-level Classification Result

Semantic segmentation is fundamentally a pixel-level classification task. For a test image with dimensions of 512×512, there are a total of 262,144 pixels to classify. The classification results can be effectively visualized using a confusion matrix. In this case, the test dataset consists of 288 images, amounting to a total of **75,497,472 pixels**.

![](README_md_files/c3e6d7e0-bab4-11ef-bac3-29376acdd9c9_20241215151820.jpeg?v=1&type=image&token=V1%3AEoMGdAYzEZibjzp9jNaJT0CmU_8AFnbrluj38r5YK14)

For this confusion matrix, below are some key measurement to evaluate the model performance. The accuracy is 95.03%, indicating most of pixel is classify correctly.

- Accuracy: 0.9503

- Precision: 0.9774

- Recall: 0.9654

- Specificity: 0.8473

- F1-Score: 0.9714

## **5.2 Segmentation Results for Target Area**

## 5.2.1 IoU Result

Accuracy measures the overall correctness of pixel classification but is susceptible to the influence of class imbalance. For instance, non-target areas (e.g., background) typically occupy the majority of pixels, allowing the model to achieve a high Accuracy score even if it fails to accurately segment the target areas.

IoU directly calculates the ratio of the intersection to the union of the predicted and ground truth areas. It provides a more intuitive reflection of the spatial consistency between the segmentation results and the actual target regions.

Within the 288 images in the test dataset, the IoU is 0.6855. Compared to high precision (0.9503) but a relatively lower IoU (0.6855), this indicates that the model's performance in predicting **target areas (e.g., landfills)** is insufficient, suggesting room for improvement in accurately delineating the boundaries of the target regions.

## **5.2.2 Examples of Landfill Segmentation Result**

### **Example 1**

Below is the segmentation result of image of "02.jpg"

1.  **Input RGB Image**:\
    This is the original input image showing an aerial or satellite view of a landscape. The image captures various elements, including roads, vegetation, and potential landfill areas.

2.  **Ground Truth Landfill Area**:\
    This is the manually labeled ground truth segmentation mask. The landfill areas are represented in **white**, while non-landfill areas are shown in **black**. This serves as the reference for evaluating the model's performance.

3.  **Predicted Landfill Area**:\
    This is the segmentation mask predicted by the trained model. Comparing this mask with the ground truth reveals the accuracy of the model in identifying and delineating landfill boundaries.

| **Image Type**             | **Image**                                                                                                                                           |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input RAB image            | ![](README_md_files/0f5aa200-bb15-11ef-bac3-29376acdd9c9_20241216024738.jpeg?v=1&type=image&token=V1%3AMswAnSBfrpOlC3W7_kLALgnONzFA4Tc3T83NLCRg65I) |
| Ground Truth Landfill Area | ![](README_md_files/19d4f9b0-bb15-11ef-bac3-29376acdd9c9_20241216024756.jpeg?v=1&type=image&token=V1%3AmewJ868MNU1hCxMPxK4E6nMcMNuGb7tvHujc_1FJWp8) |
| Predicted Landfill Area    | ![](README_md_files/1de503b0-bb15-11ef-bac3-29376acdd9c9_20241216024803.jpeg?v=1&type=image&token=V1%3ApXOEecAmhQ4IyWqj1WNBnfnBAahB-Tt259V3Km0wNgc) |

By comparing the predicted landfill mask with the ground truth, we can assess the model's segmentation performance for this image. Since the landfill in this example is distinct from the farmland, with clear and simple boundaries, the model successfully identifies and delineates the landfill areas.

The IoU for this image is **0.8202**, which aligns with our visual observation.

### **Example 2**

Below is the segmentation result of image of "265.jpg".

| **Image Type**             | **Image**                                                                                                                                           |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input RAB image            | ![](README_md_files/977ad200-bb19-11ef-bac3-29376acdd9c9_20241216032006.jpeg?v=1&type=image&token=V1%3AsQ5f7xUodPKZ4uAvE4yk2v0NNV0y90jBJYgZQeMvdYI) |
| Ground Truth Landfill Area | ![](README_md_files/9ca5c1e0-bb19-11ef-bac3-29376acdd9c9_20241216032013.jpeg?v=1&type=image&token=V1%3AcprqdQDL6gJEbSoppGPATAy3FQ67cyHdj8TBt2dgJsU) |
| Predicted Landfill Area    | ![](README_md_files/a0f37120-bb19-11ef-bac3-29376acdd9c9_20241216032020.jpeg?v=1&type=image&token=V1%3AfX37GsvDYYaaHAW2H9NS9DK0Xa8XtKIaLHNl1SaMdL0) |

The IoU for this example is **0.4370**. While the model successfully identifies the landfill and outlines its boundaries, it mistakenly classifies the construction site in the top-right corner as landfill. This error occurs because landfills and construction sites share many similar characteristics. This highlights the model's limitation in distinguishing between the two, emphasizing the need for further improvement to address this issue.

### **Example 3**

Below is the segmentation result of image of "co173.jpg".

| **Image Type**             | **Image**                                                                                                                                           |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input RAB image            | ![](README_md_files/ada94f00-bb1b-11ef-bac3-29376acdd9c9_20241216033501.jpeg?v=1&type=image&token=V1%3A-sKP_52ISJ3cdyRxPDNma_lwSyBCvFlH4_oTfZrevC8) |
| Ground Truth Landfill Area | ![](README_md_files/b21b9d40-bb1b-11ef-bac3-29376acdd9c9_20241216033508.jpeg?v=1&type=image&token=V1%3A4AP7DUdjfYc-JHn4FMj3P19zsB3PZv3NgPepcZ_dUdE) |
| Predicted Landfill Area    | ![](README_md_files/b640dcf0-bb1b-11ef-bac3-29376acdd9c9_20241216033515.jpeg?v=1&type=image&token=V1%3AuzvKB8UCz2ad03Ls5BglA6QY7O43YOXa5ihV_e9wQTM) |

This example is a clip from a village, containing farmland and houses but no landfill. The model successfully detects this, as shown in the predicted image where all pixels are black. This result demonstrates the model's ability to distinguish background elements (e.g., houses, farmland, and roads) from landfill areas.

### **5.3 Contributions and Limitations**

The results on the test dataset demonstrate that the proposed dataset and the DeepLabv3+ model can effectively identify landfills using RGB remote sensing images.

#### **Contributions Beyond Previous Studies**

1.  **Detailed Segmentation**: Compared to traditional machine learning models, DeepLabv3+ achieves more precise segmentation by capturing spatial context and fine-grained landfill boundaries.

2.  **Scalability**: This study applies deep learning to a large-scale dataset (75 million pixels), showcasing its applicability to vast geographic regions.

3.  **Automated Process**: Unlike previous studies that rely on manual feature selection, this research utilizes an **end-to-end learning framework** to identify landfill areas directly from raw imagery.

#### **Limitations and Future Directions**

1.  The model struggles to differentiate between landfills and construction sites due to their shared characteristics. Improving this capability will be an important focus for future research.

2.  While the model performs well on the test dataset with fixed image dimensions (512×512 pixels), its performance on large-scale, full landfill areas remains unknown. Evaluating the model's ability to identify entire landfills in large, real-world settings is critical for real-time applications. This is essential since the goal is to detect complete landfills in unknown areas, not just segmented parts.

Future work should address these limitations to improve the model's robustness and applicability in practical scenarios.

# **Reference:**

Fang, G. (2024). DeepLabV3Plus-Pytorch. GitHub. https\://github.com/VainF/DeepLabV3Plus-Pytorch

Nikulishyn, V., Savchyn, I., Lompas, O., & Lozynskyi, V. (2020). Applying of geodetic methods for monitoring the effects of waste-slide at Lviv municipal solid waste landfill. Environmental Nanotechnology, Monitoring & Management, 13, 100291.

Ramos, M., & Martinho, G. (2023). An assessment of the illegal dumping of construction and demolition waste. Cleaner Waste Systems, 4, 100073.

Singh, A. (2019). Remote sensing and GIS applications for municipal waste management. Journal of environmental management, 243, 22-29.

Yazdani, M., Kabirifar, K., Frimpong, B. E., Shariati, M., Mirmozaffari, M., & Boskabadi, A. (2021). Improving construction and demolition waste collection service in an urban area using a simheuristic approach: A case study in Sydney, Australia. Journal of Cleaner Production, 280, 124138. and&#x20;

Wang, J., Yong, Q., Wu, H., & Chen, R. (2022, December). Automatic Classification of Remote Sensing Images of Landfill Sites Based on Deep Learning. In International Symposium on Advancement of Construction Management and Real Estate (pp. 366-378). Singapore: Springer Nature Singapore.

Wu, H., Yong, Q., Wang, J., Lu, W., Qiu, Z., Chen, R., & Yu, B. (2024). Developing a regional scale construction and demolition waste landfill landslide risk rapid assessment approach. Waste Management, 184, 109-119.

Yong, Q., Wu, H., Wang, J., Chen, R., Yu, B., Zuo, J., & Du, L. (2023). Automatic identification of illegal construction and demolition waste landfills: A computer vision approach. Waste Management, 172, 267-277.

Zhou, L., Luo, T., Du, M., Chen, Q., Liu, Y., Zhu, Y., ... & Yang, K. (2021). Machine learning comparison and parameter setting methods for the detection of dump sites for construction and demolition waste using the google earth engine. Remote Sensing, 13(4), 787.

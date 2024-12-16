# DeepLabv3+ Model Training Workflow

The script `1_model_training.py` is designed to fine-tune the DeepLabv3+ deep learning model for identifying demolition and waste landfills in maps, as well as performing semantic segmentation to outline their boundaries.

## 1. Dataset Preparation

### Input Data:

- `images/`: Contains input images of size `(512, 512, 3)` in `.jpg` format.

- `masks/`: Contains corresponding binary segmentation masks of size `(512, 512, 1)` in `.png` format.

### Dataset Structure:

The `dataset/` directory contains `train/` and `val/` subdirectories, each with `images/` and `masks/` folders.

Ensure the file names for `images/` and `masks/` match. For example:

- dataset/train/images/001.jpg&#x20;

- dataset/train/masks/001.png

## 2. Data Generator

The `SegmentationDataGenerator` class:

- Reads and preprocesses images and masks.

- Normalizes image pixel values to `[0, 1]` and binarizes masks.

- Batches the data during training.

```python
train_generator = SegmentationDataGenerator(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
val_generator = SegmentationDataGenerator(VAL_IMAGES_DIR, VAL_MASKS_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
```

## 3. Model Definition

The script defines a DeepLabv3+ model using the following components: &#x20;

- Backbone:&#x20;

ResNet50 is used as the feature extractor, initialized with ImageNet weights.&#x20;

- ASPP Layer:&#x20;

A simplified Atrous Spatial Pyramid Pooling (ASPP) layer is added to extract multi-scale features.&#x20;

- Upsampling Layers:&#x20;

Sequential bilinear upsampling layers are used to restore the resolution to match the input size (512, 512).&#x20;

- Output Layer:&#x20;

A 1×1 convolution with a sigmoid activation produces the binary segmentation mask. &#x20;

```python
model = create_deeplabv3(input_shape=(512, 512, 3))
```

## 4. Model Compilation&#x20;

The model is compiled with: &#x20;

- Optimizer:&#x20;

Adam with a learning rate of 1e-4. Loss Function: Binary Cross-Entropy Loss.&#x20;

- Metrics:&#x20;

Accuracy and Mean Intersection over Union (IoU). &#x20;

```markup
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
```

## 5. Model Training

The model.fit function trains the model: &#x20;

- Input: Training data (train_generator) and validation data (val_generator).&#x20;

- Epochs: 25 (default in the script).&#x20;

- Batch Size: 4.&#x20;

```markup
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    verbose=1
)
```

## 6. Model Saving&#x20;

The trained model is saved in HDF5 format as deeplabv3_model.h5. &#x20;

```markup
model.save('deeplabv3_model.h5')
```

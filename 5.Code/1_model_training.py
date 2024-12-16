import tensorflow as tf

#Check is GPU available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU is available: {gpus}")
else:
    print("No GPU available, using CPU.")



import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50

# Paths to dataset
TRAIN_IMAGES_DIR = 'dataset/train/images'
TRAIN_MASKS_DIR = 'dataset/train/masks'
VAL_IMAGES_DIR = 'dataset/val/images'
VAL_MASKS_DIR = 'dataset/val/masks'

IMG_HEIGHT, IMG_WIDTH = 512, 512  # Image dimensions
BATCH_SIZE = 4
EPOCHS = 25

# --- Data Generator ---
class SegmentationDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, img_height, img_width, **kwargs):
        super().__init__(**kwargs)  # 调用父类的构造方法，传递参数
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.image_filenames = os.listdir(image_dir)


    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        masks = []

        for file_name in batch_images:
            # Load and preprocess image
            img_path = os.path.join(self.image_dir, file_name)
            image = load_img(img_path, target_size=(self.img_height, self.img_width))
            image = img_to_array(image) / 255.0  # Normalize to [0, 1]
            images.append(image)

            # Load and preprocess mask
            mask_path = os.path.join(self.mask_dir, file_name.replace('.jpg', '.png'))
            mask = load_img(mask_path, target_size=(self.img_height, self.img_width), color_mode="grayscale")
            mask = img_to_array(mask) / 255.0  # Normalize to [0, 1]
            mask = np.where(mask > 0.5, 1, 0)  # Binarize
            masks.append(mask)

        return np.array(images), np.array(masks)

# --- Load Data ---
train_generator = SegmentationDataGenerator(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
val_generator = SegmentationDataGenerator(VAL_IMAGES_DIR, VAL_MASKS_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)

# --- Define DeepLabv3+ Model ---
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def create_deeplabv3(input_shape=(512, 512, 3)):
    # Base model: ResNet50 as feature extractor
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # ASPP Layer (this is a simplified version; a true ASPP block would include parallel dilated convolutions)
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Upsample output to match 512x512 resolution
    # Assuming the output of base_model is about 16x16, we need 5 upsamplings:
    # 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 16->32
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 32->64
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 64->128
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 128->256
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)  # 256->512

    # Output layer for binary segmentation (1 channel)
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

model = create_deeplabv3()


# --- Compile Model ---
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

print(f"Model Input Shape: {model.input_shape}")
print(f"Model Output Shape: {model.output_shape}")


# --- Train Model ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    verbose=1
)

# --- Save Model ---
model.save('deeplabv3_model.h5')
print("Model training complete and saved as 'deeplabv3_model.h5'")

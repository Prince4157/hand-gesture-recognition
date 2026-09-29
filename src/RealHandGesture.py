import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os


class AdvancedHandGesture:
    def __init__(self):
        self.img_size = (160, 160)
        # Force these specific classes to ignore extra folders
        self.class_names = ['rock', 'paper', 'scissors']
        self.dataset_path = 'rps_dataset'

    def build_transfer_model(self):
        # Using MobileNetV2 as the base
        base_model = applications.MobileNetV2(input_shape=(160, 160, 3),
                                              include_top=False,
                                              weights='imagenet')
        base_model.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),  # Increased dropout for better accuracy
            layers.Dense(3, activation='softmax')  # Exactly 3 outputs
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✓ Advanced MobileNetV2 Model Built")

    def train(self):
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        # CRITICAL FIX: Explicitly set the 'classes' list
        train_gen = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='training',
            classes=self.class_names  # This ignores the 4th folder
        )

        val_gen = datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names  # This ignores the 4th folder
        )

        print(f"Training on {train_gen.samples} images...")

        self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=[ModelCheckpoint('best_real_model.keras', save_best_only=True)]
        )
        print("\n✓ Training Complete! File 'best_real_model.keras' updated.")


if __name__ == "__main__":
    trainer = AdvancedHandGesture()
    trainer.build_transfer_model()
    trainer.train()
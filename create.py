import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Step 1: Define paths to your local dataset
train_dir = r'C:\Users\DELL\OneDrive\Desktop\KOKO\dataset\train'  # Modify this to your local train data folder
test_dir = r'C:\Users\DELL\OneDrive\Desktop\KOKO\dataset\test'    # Modify this to your local test data folder

# Step 2: Set image dimensions and batch size
image_height = 150
image_width = 150
batch_size = 16  # You can reduce this if you get the OUT_OF_RANGE error
epochs = 35

# Step 3: Prepare ImageDataGenerators for data augmentation and loading images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    shear_range=0.2,  # Random shear transformations
    zoom_range=0.2,   # Random zoom transformations
    horizontal_flip=True,  # Random horizontal flips
    rotation_range=20,  # Random rotations
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2  # Vertical shift
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for the test set

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'  # Modify this to 'binary' for binary classification
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical'  # Modify this to 'binary' for binary classification
)

# Step 4: Define a 7-layer custom CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(train_set.num_classes, activation='softmax')  # Adjust based on the number of classes
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Set up EarlyStopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 7: Train the model
history = model.fit(
    train_set,
    steps_per_epoch=train_set.samples // batch_size,
    epochs=epochs,
    validation_data=test_set,
    validation_steps=test_set.samples // batch_size,
    callbacks=[early_stopping]
)

# Step 8: Save the model
model_save_path = r'C:\Users\DELL\OneDrive\Desktop\KOKO\cnn.h5'  # or .keras
model.save(model_save_path)
print(f'Model saved to: {model_save_path}')

# Step 9: Plot training & validation accuracy
plt.figure(figsize=(10, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Step 10: Generate confusion matrix
# Predict on the test set
test_set.reset()  # Ensure the test set is reset to start from the beginning
y_pred = model.predict(test_set, steps=test_set.samples // batch_size + 1, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels for the test set
y_true = test_set.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_set.class_indices.keys(), yticklabels=test_set.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

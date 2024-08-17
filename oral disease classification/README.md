# Mouth and Oral Disease Classification using InceptionResNetV2
#### ``` Please note that this implementation only includes the InceptionResNetV2 model. Due to a busy schedule this week, I was unable to implement an additional model for comparison of results.```
## Overview
This project implements a fine-tuned InceptionResNetV2 model using TensorFlow for classifying mouth and oral diseases. The model is trained on a dataset of images, augmented to enhance its performance, and evaluated for accuracy.


## Connecting to Google Drive
The project is designed to run on Google Colab, where the dataset is stored in Google Drive. Connect to Google Drive as follows:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
# List the contents of your shared drives
print(os.listdir('/content/drive/MyDrive'))
```
## Data Preparation and Augmentation
The dataset is located in Google Drive and includes training, validation, and testing sets. Data augmentation is applied to the training set to improve the model's robustness.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = '/content/drive/MyDrive/Teeth_Dataset/Training'
validation_dir = '/content/drive/MyDrive/Teeth_Dataset/Validation'
test_dir = '/content/drive/MyDrive/Teeth_Dataset/Testing'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.0]
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

test_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    shuffle=False
)
```
## Model Architecture and Fine-Tuning
The InceptionResNetV2 model is used as the base model. Additional layers are added for classification, and the base model layers are frozen initially. The model is then fine-tuned by unfreezing some layers.

```python
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load the base model
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50
)

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True

# Recompile the model for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50
)
```
## Evaluation
The model's performance is evaluated on the test set, and a confusion matrix and classification report are generated to provide detailed performance metrics.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy}')

# Get true labels and predictions
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = y_pred.argmax(axis=-1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
print(classification_report(y_true, y_pred_classes, target_names=train_generator.class_indices.keys()))
```
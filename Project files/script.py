import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

train_csv = pd.read_csv("Training_set.csv")
test_csv = pd.read_csv("Testing_set.csv")

print("Train CSV shape:", train_csv.shape)
print("Test CSV shape:", test_csv.shape)
print(train_csv.head())

# ✅ Step 3: Load & Resize Images
image_size = (224, 224)  # for VGG16
X = []
y = []

# Loop through each row in CSV
for index, row in train_csv.iterrows():
    img_path = os.path.join("train", row['filename'])  # train/Image_1.jpg
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        X.append(img)
        y.append(row['label'])
    else:
        print(f"❌ File not found: {img_path}")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("✅ Loaded images:", X.shape)
print("✅ Labels:", y.shape)
# ✅ Step 4: Encode Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # ADONIS -> 0, MONARCH -> 1, etc.

# One-hot encode labels
y_categorical = to_categorical(y_encoded)

# Split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42)

print("✅ X_train:", X_train.shape)
print("✅ y_train:", y_train.shape)
print("✅ X_val:", X_val.shape)
print("✅ y_val:", y_val.shape)

# ✅ Step 5: Load VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
num_classes = y_categorical.shape[1]  # Automatically get correct class count
predictions = Dense(num_classes, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save("butterfly_model_vgg16.h5")
print("✅ Model trained and saved successfully.")
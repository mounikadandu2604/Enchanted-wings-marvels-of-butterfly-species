import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ✅ 1. Load the best trained model
model = load_model("Vgg16_model.h5")

# ✅ 2. Load the CSV again to get label names
train_csv = pd.read_csv("Training_set.csv")
labels = train_csv['label'].unique()
labels.sort()
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# ✅ 3. Load & prepare the test image
img_path = "test/Image_123.jpg"  # change to your image
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# ✅ 4. Predict the class
prediction = model.predict(img)
predicted_class = np.argmax(prediction)
class_label = label_encoder.inverse_transform([predicted_class])[0]

print("🦋 Predicted Butterfly Class:", class_label)
from keras.models import load_model, Model
from keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np

# Patch DepthwiseConv2D to ignore "groups" argument
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

custom_objects = {"DepthwiseConv2D": PatchedDepthwiseConv2D}

np.set_printoptions(suppress=True)

# Load the model with custom_objects
model = load_model("keras_Model.h5", compile=False, custom_objects=custom_objects, safe_mode=False)

# Load labels
class_names = open("labels.txt", "r").readlines()

# Image preprocessing
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open("snip_20250927_170629_577041.png").convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
data[0] = (image_array.astype(np.float32) / 127.5) - 1

# Predict
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]

print(f"Class: {class_name}, Confidence Score: {confidence_score:.4f}")

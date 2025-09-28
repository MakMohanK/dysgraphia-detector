from keras.models import load_model, Model
import cv2
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D

# 🔹 Patch DepthwiseConv2D
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)

custom_objects = {"DepthwiseConv2D": PatchedDepthwiseConv2D}
np.set_printoptions(suppress=True)

# 🔹 Load model
raw_model = load_model(
    "./model/keras_model.h5",
    compile=False,
    custom_objects=custom_objects,
    safe_mode=False
)

# 🔹 If model has multiple inputs, take only the first one
if isinstance(raw_model.input, (list, tuple)):
    model = Model(inputs=raw_model.input[0], outputs=raw_model.output)
else:
    model = raw_model

# 🔹 Load labels
with open("./model/labels.txt", "r") as f:
    class_names = f.readlines()

def predict_from_image_path(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, f"{confidence_score * 100:.2f}"

print(predict_from_image_path("ml/Camera_ai/snips/snip_20250927_170629_577041.png"))

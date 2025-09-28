from keras.models import load_model

# Load your old Keras 2 / H5 model
model = load_model("keras_Model.h5", compile=False)

# Save it in the new Keras 3 format (.keras)
model.save("keras_model_converted.keras")

print("Model converted successfully!")
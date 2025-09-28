# convert_to_tflite.py
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('../models/model_lstm.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset generator (use sample windows from training set)
X_rep = np.load('X_seq.npy')[:200].astype(np.float32)
def rep_gen():
    for i in range(X_rep.shape[0]):
        sample = X_rep[i:i+1]
        yield [sample]

converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
open('../models/model.tflite','wb').write(tflite_model)
print("Saved ../models/model.tflite")

# realtime_infer.py
import serial, numpy as np, time
try:
    import tflite_runtime.interpreter as tflite
except:
    import tensorflow as tf
    tflite = tf.lite

MODEL = '../models/model.tflite'
PORT = 'COM3'  # change
BAUD = 115200
WINDOW = 200

# load model
interpreter = tflite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

ser = serial.Serial(PORT, BAUD, timeout=1)
buf = []

while True:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if not line: continue
    fields = line.split(',')
    if len(fields) < 9: continue
    # pick ax..p2 numeric columns (skip timestamp)
    vals = [float(x) for x in fields[1:9]]
    buf.append(vals)
    if len(buf) >= WINDOW:
        window_arr = np.array(buf[-WINDOW:]).astype(np.float32)
        inp = np.expand_dims(window_arr, axis=0)
        # scale/normalize as you trained (apply same normalization)
        inp = inp.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        print("Pred:", out)
    time.sleep(0.01)

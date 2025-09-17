# Dysgraphia Detection Using Seeed Studio + MPU6050 + Pressure Sensors

### Folder Structure
~~~Python
dysgraphia-detector/
├─ hardware/
│  └─ wiring_diagram.txt
├─ firmware/
│  ├─ mpu_pressure_logger.ino
│  └─ tflite_micro_inference.ino   # skeleton
├─ pc_logger/
│  └─ serial_logger.py
├─ ml/
│  ├─ feature_extraction.py
│  ├─ build_dataset.py
│  ├─ train_random_forest.py
│  ├─ train_lstm.py
│  ├─ convert_to_tflite.py
│  └─ realtime_infer.py
├─ models/
│  ├─ model_rf.pkl
│  ├─ model_lstm.h5
│  └─ model.tflite
├─ requirements.txt
└─ README.md
~~~

### Hardware Connections
#### MPU6050 (I²C)
<img src="img\mpu_interfacing.jpg" alt="XIAO RP2040 Board" width="400">

~~~ 
VCC → 3.3V
GND → GND
SCL → D5 (SCL pin of Seeed board)
SDA → D4 (SDA pin of Seeed board)
~~~

#### Pressure Sensors (Analog)
<img src="img\image.png" alt="XIAO RP2040 Board" width="400">

~~~
Suppose you’re using force-sensitive resistors (FSR) or piezo pressure sensors.
Sensor 1 output → A0 (Analog pin)
Sensor 2 output → A1 (Analog pin)
One end → 3.3V
Other end → Voltage divider resistor (10kΩ) → GND
~~~



### 📌 Project Overview

This project detects dysgraphia (writing disorder) by analyzing handwriting motion and pressure patterns.
We use:
Seeed Studio board (XIAO SAMD21 / RP2040 / ESP32)
MPU6050 (accelerometer + gyroscope)
Two pressure sensors (FSR/Piezo)
Data is collected while a user writes, processed in Python, and classified with an ML model.

###  🔧 Hardware Requirements
Seeed Studio XIAO (or similar)
MPU6050 IMU
2× Pressure sensors (FSR / Piezo + resistors)
Breadboard + jumper wires
USB cable

###  🖥️ Software Setup
1. Arduino IDE
Install Arduino IDE
Install libraries:
MPU6050
Wire

2. Python Environment
pip install pandas numpy scikit-learn matplotlib

### 📜 Step-by-Step Instructions
Step 1: Collect Data

Upload Arduino sketch (data_logger.ino) to Seeed board.
Open Serial Monitor (baud 115200).
Save CSV data:
~~~
ax,ay,az,gx,gy,gz,pressure1,pressure2
0.12,-0.03,9.80,0.5,-0.2,1.1,512,300
~~~

Step 2: Preprocess Data (Python)
Run preprocess.py
Cleans data, normalizes sensor readings, extracts features.

Step 3: Train ML Model
Run train_model.py
Splits dataset (train/test).
Trains a classifier (RandomForest / XGBoost).
Saves model as dysgraphia_model.pkl.

Step 4: Test Prediction
Run predict.py with new data.
Example:
python predict.py sample.csv

Output:
Prediction: Dysgraphia detected ✅

Step 5: (Optional) Edge Deployment
Convert trained model → TensorFlow Lite Micro.
Upload to Seeed board for real-time inference.

### ✅ Next Steps

Collect labeled datasets (Normal, Dysgraphia).
Tune ML models for higher accuracy.
Deploy model to hardware.